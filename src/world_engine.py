from typing import Dict, Optional, Set, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from dataclasses import dataclass, field

from owl_wms.models.world import WorldModel
from owl_wms.nn.kv_cache import StaticKVCache

from world_engine.ae import InferenceAE


# Global torch optimizations
torch._dynamo.config.recompile_limit = 64
torch.set_float32_matmul_precision("medium")  # low: bf16, medium: tf32, high: fp32

# fix graph break:
torch._dynamo.config.capture_scalar_outputs = True


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) position


@dataclass
class InferenceConfig:
    quantize: bool = True  # False: bf16, true: w8a8
    # TODO: use model config scheduler sigmas
    # scheduler_sigmas: Optional[List[float]] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0])
    # noise_prev: float = 0.0  # always 0 due to self forcing


@dataclass
class OptimizationConfig:
    pass  # TODO
"""
TODO: quantization options
- bf16
- a8w8: Float8DynamicActivationFloat8WeightConfig
"""


class WorldEngine:
    def __init__(
        self,
        model_uri: str,
        inference_config: Optional[InferenceConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        model_config_overrides: Optional[Dict] = None,
        device=None,
        dtype=torch.bfloat16,
    ):
        # Meta
        self.device, self.dtype = device, dtype
        inference_config = inference_config or InferenceConfig()
        self.model_cfg = WorldModel.load_config(model_uri)

        # TODO: remove these hardcoding hacks:
        if not hasattr(self.model_cfg, "n_buttons"):
            self.model_cfg.n_buttons = self.model_cfg.n_controller_inputs - 2
        self.model_cfg.mlp_gradient_checkpointing = getattr(self.model_cfg, "mlp_gradient_checkpointing", False)

        if model_config_overrides:
            self.model_cfg.merge_with(model_config_overrides)

        # Model
        self.vae = InferenceAE.from_pretrained(model_uri, device=device, dtype=dtype)
        # self.prompt_encoder = PromptEncoder("google/umt5-xl").to(device).eval()  # TODO: dont hardcode
        self.model = WorldModel.from_pretrained(model_uri, cfg=self.model_cfg).to(device=device, dtype=dtype).eval()

        # TODO: clean up quantization logic
        if inference_config.quantize:
            self.quantize(self.model)

        # Inference Scheduler
        self.scheduler_sigmas = torch.tensor(self.model_cfg.scheduler_sigmas, device=device, dtype=dtype)

        pH, pW = getattr(self.model_cfg, "patch", [1, 1])
        self.frm_shape = 1, 1, self.model_cfg.channels, self.model_cfg.height * pH, self.model_cfg.width * pW

        # State
        self.kv_cache = StaticKVCache(self.model_cfg, max_seq_len=512, batch_size=1, dtype=dtype).to(device)
        self.frame_ts = torch.tensor([[0]], dtype=torch.long, device=device)
        self.reset()

    def quantize(self, model):
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow

        def is_bf16_linear(mod: nn.Module, _: str) -> bool:
            weight = getattr(mod, "weight", None)
            return isinstance(mod, nn.Linear) and getattr(weight, "dtype", None) is torch.bfloat16

        quant_cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        quantize_(
            self.model,
            quant_cfg,
            filter_fn=is_bf16_linear,
            device=self.device,
        )

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        self.frame_ts = self.frame_ts * 0
        self.kv_cache = StaticKVCache(self.model_cfg, max_seq_len=512, batch_size=1, dtype=self.dtype).to(self.device)

    @torch.inference_mode()
    @torch.compile
    def _push_frame_state(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        button = x.new_zeros(1, 1, self.model_cfg.n_buttons)
        if ctrl.button:
            idx = x.new_tensor(list(ctrl.button), dtype=torch.long)
            button.index_fill_(-1, idx, 1.0)
        mouse = x.new_tensor(ctrl.mouse, dtype=self.dtype)[None, None]
        inputs = {
            "button": button,
            "mouse": mouse,
            "frame_timestamp": self.frame_ts.clone(),
        }
        self.frame_ts = self.frame_ts + 1
        return inputs

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: CtrlInput = None):
        assert img.dtype == torch.uint8, img.dtype
        # push frame inputs w/ clean latent
        x = self.vae.encode(img).unsqueeze(1)
        inputs = self._push_frame_state(x=x, ctrl=ctrl)
        self.kv_cache, _ = self.denoise_frame(x, inputs, self.kv_cache, denoise=False)
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        # prepare frame inputs + random N latent
        x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
        inputs = self._push_frame_state(x=x, ctrl=ctrl)
        self.kv_cache, x = self.denoise_frame(x, inputs, self.kv_cache)
        with torch.amp.autocast('cuda', torch.bfloat16):
            return (self.vae.decode(x) if return_img else x)

    def set_prompt(self, prompt: str, timestamp: float = 0.0):
        """Apply text conditioning for T2V"""
        import warnings
        warnings.warn("Not Implemented")

    @torch.inference_mode()
    @torch.compile(fullgraph=True, mode="max-autotune", dynamic=False)
    def denoise_frame(self, x, state: Dict[str, Tensor], kv_cache, denoise: bool = True):
        """Advance state, generate new frame, provide updated kv_cache, and denoised x"""
        if denoise:
            kv_cache.set_frozen(True)
            x = self._denoise_pass(x, state, kv_cache)
        kv_cache.set_frozen(False)
        kv_cache = self._update_kv_pass(x, state, kv_cache)
        return kv_cache, x.squeeze(1)

    def _denoise_pass(self, x, state: Dict[str, Tensor], kv_cache):
        sigma = x.new_empty((x.size(0), x.size(1)))
        for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
            v = self.model(x, sigma.fill_(step_sig), **state, kv_cache=kv_cache)
            x = x + step_dsig * v
        return x

    def _update_kv_pass(self, x, state: Dict[str, Tensor], kv_cache):
        self.model(x, x.new_zeros((x.size(0), x.size(1))), **state, kv_cache=kv_cache)
        return kv_cache


# TODO
# - Push inference config overrides to hub (if reasonable, set an inference config in training)
# - RoPE for inference
