import argparse
import time
from dataclasses import dataclass
from typing import List, Optional
import gc

import torch


@dataclass
class BenchmarkResult:
    config: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    memory_mb: float
    n_iters: int


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def print_env_info():
    """Print environment and hardware information."""
    import platform

    print("\n" + "=" * 60)
    print("ENVIRONMENT INFO")
    print("=" * 60)

    print(f"\ntorch:        {torch.__version__}")
    print(f"torch.cuda:   {torch.version.cuda}")

    print(f"\nOS:   {platform.system()} {platform.release()} ({platform.machine()})")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"\nGPU:  {props.name}")
        print(f"      SM capability: {props.major}.{props.minor}")
        print(f"      Total memory: {props.total_memory / 1e9:.1f} GB")

        is_blackwell = props.major == 12 and props.minor in (0, 1)
        print(f"      Blackwell (FP4 capable): {'Yes' if is_blackwell else 'No'}")
    else:
        print("\nGPU:  None (CUDA not available)")

    try:
        from world_engine.fp4_linear import HAS_FP4

        print(f"\nFlashInfer FP4: {'Available' if HAS_FP4 else 'Not available'}")
    except ImportError:
        print("\nFlashInfer FP4: Not available (import failed)")


def create_engine(model_uri: str, use_fp4: bool = False):
    """Create a WorldEngine with optional FP4 patching."""
    from world_engine import WorldEngine
    from world_engine.patch_model import patch_linear_to_fp4

    engine = WorldEngine(model_uri, quant=None, apply_patches=False, device="cuda")

    if use_fp4:
        patched_count = patch_linear_to_fp4(engine.model)
        print(f"  Patched {patched_count} layers to FP4Linear")

    return engine


def warmup_engine(engine, n_warmup: int = 3):
    """Warmup the engine with a few generations."""
    x = torch.randn(engine.frm_shape, device="cuda", dtype=torch.bfloat16)
    engine.reset()
    for _ in range(n_warmup):
        engine.gen_frame(x=x, return_img=False)
    torch.cuda.synchronize()


def benchmark_generation(
    engine, n_iters: int, use_cuda_graph: bool = True
) -> List[float]:
    """Benchmark frame generation, returning per-iteration times in ms."""
    engine.reset()
    engine.gen_frame(return_img=False)
    torch.cuda.synchronize()

    x = torch.randn(engine.frm_shape, device="cuda", dtype=torch.bfloat16)

    if use_cuda_graph:
        # First call captures the graph
        engine.gen_frame_graph(x=x, return_img=False)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(n_iters):
            start = time.perf_counter()
            engine.gen_frame_graph(x=x, return_img=False)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)
    else:
        times_ms = []
        for _ in range(n_iters):
            start = time.perf_counter()
            engine.gen_frame(return_img=False)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)

    return times_ms


def run_benchmark(
    config: str,
    model_uri: str,
    n_warmup: int,
    n_iters: int,
    use_cuda_graph: bool = True,
) -> Optional[BenchmarkResult]:
    """Run benchmark for a single configuration."""
    print(f"\n{'─' * 50}")
    print(f"Config: {config}")
    print(f"{'─' * 50}")

    reset_gpu_memory()

    try:
        use_fp4 = config == "fp4"

        if use_fp4:
            try:
                from world_engine.fp4_linear import HAS_FP4

                if not HAS_FP4:
                    print("  ⚠ FP4 not available (no FlashInfer)")
                    return None
            except ImportError as e:
                print(f"  ✗ FP4 import failed: {e}")
                return None

        print("  Loading model...")
        engine = create_engine(model_uri, use_fp4=use_fp4)

        print(f"  Warming up ({n_warmup} iterations)...")
        warmup_engine(engine, n_warmup)

        reset_gpu_memory()

        mode = "CUDA graph" if use_cuda_graph else "eager"
        print(f"  Benchmarking ({n_iters} iterations, {mode})...")
        times_ms = benchmark_generation(engine, n_iters, use_cuda_graph)

        memory_mb = get_gpu_memory_mb()

        import statistics

        result = BenchmarkResult(
            config=config,
            mean_ms=statistics.mean(times_ms),
            std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
            memory_mb=memory_mb,
            n_iters=n_iters,
        )

        print(f"  ✓ Mean: {result.mean_ms:.2f} ms (±{result.std_ms:.2f})")
        print(f"  ✓ Memory: {result.memory_mb:.0f} MB")

        del engine
        torch.cuda.empty_cache()
        gc.collect()

        return result

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def print_results_table(results: List[BenchmarkResult]):
    """Print a formatted comparison table."""
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    baseline = next((r for r in results if r.config == "bf16"), None)

    print(
        f"\n{'Config':<10} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} "
        f"{'Max (ms)':<10} {'Memory (MB)':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    for r in results:
        speedup = ""
        if baseline and r.config != "bf16":
            speedup = f"{baseline.mean_ms / r.mean_ms:.2f}x"

        print(
            f"{r.config:<10} {r.mean_ms:<12.2f} {r.std_ms:<10.2f} {r.min_ms:<10.2f} "
            f"{r.max_ms:<10.2f} {r.memory_mb:<12.0f} {speedup:<10}"
        )

    print("-" * 80)

    if len(results) > 1:
        fastest = min(results, key=lambda r: r.mean_ms)
        print(f"\nFastest: {fastest.config} ({fastest.mean_ms:.2f} ms)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BF16 vs FP4 quantization")
    parser.add_argument(
        "--model-uri",
        "-m",
        default="OpenWorldLabs/Codbuland",
        help="Model URI or local path",
    )
    parser.add_argument(
        "--n-warmup",
        "-w",
        type=int,
        default=30,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--n-iters",
        "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph capture (use eager mode)",
    )
    parser.add_argument(
        "--bf16-only",
        action="store_true",
        help="Only run bf16 benchmark",
    )
    parser.add_argument(
        "--fp4-only",
        action="store_true",
        help="Only run fp4 benchmark",
    )

    args = parser.parse_args()

    if args.bf16_only:
        configs = ["bf16"]
    elif args.fp4_only:
        configs = ["fp4"]
    else:
        configs = ["bf16", "fp4"]

    print_env_info()

    use_cuda_graph = not args.no_cuda_graph

    print("\n" + "=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"\nModel:       {args.model_uri}")
    print(f"Configs:     {', '.join(configs)}")
    print(f"Warmup:      {args.n_warmup} iterations")
    print(f"Benchmark:   {args.n_iters} iterations")
    print(f"CUDA graph:  {use_cuda_graph}")

    results = []
    for config in configs:
        result = run_benchmark(
            config=config,
            model_uri=args.model_uri,
            n_warmup=args.n_warmup,
            n_iters=args.n_iters,
            use_cuda_graph=use_cuda_graph,
        )
        if result:
            results.append(result)

    print_results_table(results)


if __name__ == "__main__":
    main()
