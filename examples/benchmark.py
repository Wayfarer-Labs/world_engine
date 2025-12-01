import pyperf
import time
from world_engine import WorldEngine, CtrlInput

import warnings
warnings.filterwarnings("ignore")


# TODO
# - benchmark encode img
# - benchmark encode prompt
# - isolated benchmark: decode image time

# TODO:
"""
world_engine version: <version / commit>

Hardware:
- details
- details

Benchmarks:
- benchmark
- benchmark
"""


def gen_n_frames(engine, n_frames):
    for _ in range(n_frames):
        engine.gen_frame()


def time_ar_rollout(loops, engine, n_frames):
    start = time.perf_counter()
    for _ in range(loops):
        engine.reset()                     # not timed per-iteration if you like
        gen_n_frames(engine, n_frames)     # core work
    end = time.perf_counter()
    return end - start                    # pyperf divides by loops for you


def run_benchmark(model_uri: str = "OpenWorldLabs/CoDCtl-Causal", device: str = "cuda") -> None:
    engine = WorldEngine(model_uri, device=device, model_config_overrides={"n_frames": 512})

    # Warmup torch compilation
    for _ in range(3):
        engine.gen_frame()

    runner = pyperf.Runner(processes=1)

    for n_frames in [1, 4, 16, 64, 256]:
        runner.bench_time_func(
            f"AR Rollout n_frames={n_frames}",
            time_ar_rollout,
            engine,
            n_frames,
            inner_loops=1,
        )


if __name__ == "__main__":
    run_benchmark()
