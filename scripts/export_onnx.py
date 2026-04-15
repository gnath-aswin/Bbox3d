import torch
import argparse
import os
import onnx

from src.model import PointNetBBox


def export_onnx(model_path, output_path, device="cpu"):
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "model.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model
    model = PointNetBBox()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(" Model loaded")

    # Dummy input (Point cloud sampling)
    dummy_input = torch.randn(1, 5120, 3).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,  # modern stable opset
        do_constant_folding=True,
        input_names=["points"],
        output_names=["center", "size", "yaw"],
        dynamic_axes={
            "points": {0: "batch_size"},
            "center": {0: "batch_size"},
            "size": {0: "batch_size"},
            "yaw": {0: "batch_size"},
        },
        dynamo=False,  # avoid new exporter instability
        verbose=False,
    )

    print(f"ONNX model saved to: {output_path}")

    # Validate ONNX model
    print("Validating ONNX model...")
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)

    print("ONNX model is valid!")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to PyTorch .pth model")
    parser.add_argument(
        "--output",
        default="deployment/model.onnx",
        help="Output ONNX file or directory",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")

    args = parser.parse_args()

    export_onnx(args.model, args.output, args.device)
