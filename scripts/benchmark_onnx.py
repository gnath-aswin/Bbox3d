import onnxruntime as ort
import numpy as np
import time
import argparse


def benchmark_onnx(model_path, n_runs=100, device="cpu"):
    print(f"Loading ONNX model: {model_path}")

    # Choose provider
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"Input: {input_name}")
    print(f"Outputs: {output_names}")
    print(f"Providers: {session.get_providers()}")

    # Dummy input (match training shape)
    dummy_input = np.random.randn(1, 5120, 3).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    print("Running benchmark...")
    start = time.time()

    for _ in range(n_runs):
        outputs = session.run(None, {input_name: dummy_input})

    end = time.time()

    avg_time = (end - start) / n_runs * 1000

    print(f"Results:")
    print(f"Average inference time: {avg_time:.3f} ms")
    print(f"FPS: {1000.0 / avg_time:.2f}")

    # 🔍 Sanity check outputs
    print("\nOutput shapes:")
    for name, out in zip(output_names, outputs):
        print(f"{name}: {out.shape}")

    return avg_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deployment/model.onnx")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    benchmark_onnx(args.model, args.runs, args.device)
