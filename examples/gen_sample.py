import argparse
import cv2
from world_engine import WorldEngine
import time


def gen_vid(
    model_uri: str,
    output_path: str,
    n_frames: int,
    use_fp4: bool = False,
    use_cuda_graph: bool = False,
):
    print(f"Loading model: {model_uri}")
    engine = WorldEngine(model_uri, device="cuda", apply_patches=True)

    if use_fp4:
        from world_engine.patch_model import patch_linear_to_fp4

        n_patched = patch_linear_to_fp4(engine.model, ["emb", "norm", "patch"])
        # n_patched += patch_linear_to_fp4(engine.vae.ae_model)
        print(f"Patched {n_patched} layers to FP4Linear")

    # Select generation method
    if use_cuda_graph:
        import torch

        print("Using CUDA graph mode")
        # IMPORTANT: Must warmup before CUDA graph capture
        # 1. Reset state
        engine.reset()
        # 2. Run one eager frame to populate kv cache
        engine.gen_frame(return_img=False)
        torch.cuda.synchronize()
        # 3. First graph call captures (with random noise), subsequent replay
        engine.gen_frame_graph(return_img=False)  # x=None generates new noise
        torch.cuda.synchronize()
        print("  CUDA graph captured")

        # Generate with graph (decode VAE outside graph)
        # x=None means gen_frame_graph generates NEW random noise each frame
        def gen_fn():
            latent = engine.gen_frame_graph(return_img=False)  # Fresh noise each call
            return engine.vae.decode(latent)

    else:
        print("Using eager mode")

        def gen_fn():
            return engine.gen_frame()

    print(f"Generating {n_frames} frames...")
    writer = None

    _ = gen_fn()
    torch.cuda.synchronize()

    start = time.time()

    frames = []

    for i in range(n_frames):
        frame = gen_fn()  # RGB -> BGR for OpenCV
        frames.append(frame)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{n_frames}")

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f} seconds")

    for frame in frames:
        writer = writer or cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (frame.shape[1], frame.shape[0]),
        )
        writer.write(frame.cpu().numpy()[:, :, ::-1])

    writer.release()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sample video")
    parser.add_argument(
        "--model",
        "-m",
        default="OpenWorldLabs/CoDCtl-Causal-Flux-SelfForcing",
        help="Model URI (HuggingFace or local path)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="out.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--frames",
        "-n",
        type=int,
        default=240,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--fp4",
        action="store_true",
        help="Apply FP4 patching (Blackwell GPUs)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Use CUDA graph for faster generation",
    )
    args = parser.parse_args()

    gen_vid(
        model_uri=args.model,
        output_path=args.output,
        n_frames=args.frames,
        use_fp4=args.fp4,
        use_cuda_graph=args.cuda_graph,
    )


if __name__ == "__main__":
    main()
