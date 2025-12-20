import torch
import torch.nn as nn
from typing import Optional

try:
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout

    HAS_FP4 = True
except ImportError:
    HAS_FP4 = False


QUANTS = [None, "w8a8", "nvfp4"]


class FP8W8A8Linear(nn.Module):
    __constants__ = ("in_features", "out_features")

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features

        f8 = torch.float8_e4m3fn
        inv = 1.0 / float(torch.finfo(f8).max)
        self._inv = inv

        w = lin.weight.detach()
        ws = (w.abs().amax() * inv).clamp_min(1e-8).float()      # 0-d
        wf8 = (w / ws.to(w.dtype)).to(f8).contiguous()            # row-major
        self.register_buffer("wT", wf8.t())                       # col-major view (no contiguous)
        self.register_buffer("ws", ws)

        if lin.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", lin.bias.detach().to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        x2 = x.reshape(-1, s[-1])

        xs = (x2.abs().amax() * self._inv).clamp_min(1e-8).float()          # 0-d
        xf8 = (x2 / xs.to(x2.dtype)).to(torch.float8_e4m3fn).contiguous()

        y = torch._scaled_mm(
            xf8, self.wT, xs, self.ws,
            bias=self.bias, out_dtype=torch.float16, use_fast_accum=True
        )
        return y.reshape(*s[:-1], self.out_features).to(x.dtype)


class FP4Linear(nn.Module):
    """FP4 Linear layer using FlashInfer's NVFP4 quantization."""

    FP4_AMAX = 6.0
    FP8_AMAX = 448.0

    def __init__(self, lin: nn.Linear):
        super().__init__()

        if not HAS_FP4:
            raise RuntimeError(
                "FlashInfer with FP4 support not found. Install with:\n"
                "  pip install flashinfer"
            )

        self.in_features = lin.in_features
        self.out_features = lin.out_features

        # Check alignment requirements for NVFP4 TMA
        if self.in_features % 32 != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by 32 "
                f"for NVFP4 TMA alignment"
            )

        if self.out_features % 32 != 0:
            raise ValueError(
                f"out_features ({self.out_features}) must be divisible by 32 "
                f"for NVFP4 TMA alignment"
            )

        # Store weight from original linear layer
        self.weight = nn.Parameter(lin.weight.detach().clone())

        # Cached FP4 weight and scales (populated on first forward)
        self._weight_fp4: Optional[torch.Tensor] = None
        self._weight_scales: Optional[torch.Tensor] = None
        self._weight_global_sf: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._dummy_scale: Optional[torch.Tensor] = None

    def _quantize_weight(self, device: torch.device):
        """Quantize and cache weight to FP4 on first call."""
        if self._weight_fp4 is None:
            # Dummy scale for activations (no global scaling)
            self._dummy_scale = torch.full(
                (1,), 1.0, device=device, dtype=torch.float32
            )
            # Convert weight to BF16 for quantization
            # Weight shape: [N, K] = [out_features, in_features]
            weight_bf16 = self.weight.to(torch.bfloat16).to(device)

            # Compute global scale factor: (FP8_AMAX * FP4_AMAX) / weight_amax
            weight_amax = weight_bf16.float().abs().nan_to_num().max()
            self._weight_global_sf = (1.0) / weight_amax

            # Alpha rescales output: 1.0 / weight_global_sf
            self._alpha = 1.0 / (self._weight_global_sf * self._dummy_scale)

            # Quantize using FlashInfer with 128x4 layout for cutlass backend
            self._weight_fp4, self._weight_scales = nvfp4_quantize(
                weight_bf16,
                self._weight_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using FP4 quantization and FlashInfer GEMM.

        Args:
            x: Input tensor of shape [B, in_features] in BF16 or FP8 format

        Returns:
            Output tensor of shape [B, out_features] in BF16 format
        """
        # Ensure weights are quantized and cached
        self._quantize_weight(x.device)

        x_flat = x.reshape(-1, x.shape[-1])

        # Quantize input with scale=1.0 (per-block scales only, no global scaling)
        x_fp4, x_sf = nvfp4_quantize(
            x_flat.to(torch.bfloat16).contiguous(),
            self._dummy_scale,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )

        # Run FP4 GEMM with cutlass backend
        # mm_fp4 expects b as (k, n) col-major, so transpose weight
        return mm_fp4(
            x_fp4,
            self._weight_fp4.T,  # [N, K/2] -> [K/2, N]
            x_sf,
            self._weight_scales.T,  # Transpose scales too
            self._alpha,  # Rescale output by 1/weight_global_sf
            out_dtype=torch.bfloat16,
            backend="cutlass",
        ).reshape(x.shape[:-1] + (-1,))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


def quantize_model(model: nn.Module, quant: str):
    if quant is None:
        return model

    if quant == "nvfp4":
        return _quantize_model_fp4(model)

    def eligible(m: nn.Module) -> bool:
        w = getattr(m, "weight", None)
        if not (isinstance(m, nn.Linear) and getattr(w, "dtype", None) is torch.bfloat16):
            return False
        o, k = w.shape
        return (o % 16 == 0) and (k % 16 == 0)

    new_linear = {
        "w8a8": FP8W8A8Linear,
    }[quant]

    for name, child in model.named_children():
        setattr(model, name, new_linear(child)) if eligible(child) else quantize_model(child, quant)
    return model


def _quantize_model_fp4(model: nn.Module, skip_names: list[str] = None) -> nn.Module:
    """
    Replace all nn.Linear layers in the model with FP4Linear.

    This patch:
    1. Finds all nn.Linear modules with compatible dimensions (divisible by 32)
    2. Creates FP4Linear replacements with copied weights
    3. Replaces them in-place using set_submodule

    Args:
        model: The model to patch (should be on device and in bf16)
        skip_names: Optional list of substrings to skip (e.g., ["embed", "head"])

    Returns:
        The patched model
    """
    skip_names = skip_names or []

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue

        # Skip if already patched
        if isinstance(mod, FP4Linear):
            continue

        # Skip based on name patterns
        if any(skip in name for skip in skip_names):
            continue

        # Check alignment requirements (both dims must be divisible by 32)
        if mod.in_features % 32 != 0 or mod.out_features % 32 != 0:
            continue

        # Skip layers with bias (FP4Linear doesn't support bias)
        if mod.bias is not None:
            continue

        # Create new FP4 linear layer and replace in model
        new_linear = FP4Linear(mod)
        model.set_submodule(name, new_linear)

    return model


# Alias for backwards compatibility
patch_linear_to_fp4 = _quantize_model_fp4
