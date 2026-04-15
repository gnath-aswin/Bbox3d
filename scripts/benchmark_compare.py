import torch
import onnxruntime as ort
import numpy as np
import time
import argparse

from src.model import PointNetBBox


def load_pytorch(model_path, device):
    model = PointNetBBox().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_onnx(onnx_path, device):
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def benchmark_torch(model, dummy, device, n_runs):
    # Warmup
    for _ in range(10):
        _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(n_runs):
        out = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    return (end - start) / n_runs * 1000, out


def benchmark_onnx(session, dummy_np, n_runs):
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_np})

    start = time.time()

    for _ in range(n_runs):
        out = session.run(None, {input_name: dummy_np})

    end = time.time()
    return (end - start) / n_runs * 1000, out


def compare_outputs(torch_out, onnx_out):
    torch_out = [o.detach().cpu().numpy() for o in torch_out]

    errors = {}
    names = ["center", "size", "yaw"]

    for name, t, o in zip(names, torch_out, onnx_out):
        abs_diff = np.abs(t - o)
        rel_diff = abs_diff / (np.abs(t) + 1e-6)

        errors[name] = {
            "mean_abs": abs_diff.mean(),
            "max_abs": abs_diff.max(),
            "mean_rel": rel_diff.mean(),
            "max_rel": rel_diff.max(),
        }

    return errors


def main(args):
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"🚀 Device: {device}")

    # Load models
    torch_model = load_pytorch(args.model, device)
    onnx_session = load_onnx(args.onnx, device)

    # Dummy input
    dummy_torch = torch.randn(1, 5120, 3).to(device)
    dummy_np = dummy_torch.cpu().numpy().astype(np.float32)

    # Benchmark
    print("\n⚡ Benchmarking PyTorch...")
    torch_time, torch_out = benchmark_torch(torch_model, dummy_torch, device, args.runs)

    print("⚡ Benchmarking ONNX...")
    onnx_time, onnx_out = benchmark_onnx(onnx_session, dummy_np, args.runs)

    # Compare outputs
    print("\n🎯 Comparing outputs...")
    errors = compare_outputs(torch_out, onnx_out)

    # Print results
    print("\n📊 RESULTS")
    print("-" * 40)
    print(f"PyTorch Time: {torch_time:.3f} ms")
    print(f"ONNX Time:    {onnx_time:.3f} ms")
    print(f"Speedup:      {torch_time / onnx_time:.2f}x")

    print("\n🔍 Numerical Differences:")
    for k, v in errors.items():
        print(f"\n{k}:")
        print(f"  mean_abs: {v['mean_abs']:.6f}")
        print(f"  max_abs:  {v['max_abs']:.6f}")
        print(f"  mean_rel: {v['mean_rel']:.6f}")
        print(f"  max_rel:  {v['max_rel']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="PyTorch model path")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    main(args)
