"""ONNX export and parity verification for the selected causal model."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def export_model(model: torch.nn.Module, output_path: Path, feature_count: int) -> None:
    """Export a model with dynamic batch and fixed 10-epoch feature input."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 10, feature_count, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        opset_version=17,
        input_names=["sequences"],
        output_names=["probability"],
        dynamic_axes={"sequences": {0: "batch"}, "probability": {0: "batch"}},
    )


def verify_onnx_probabilities(model: torch.nn.Module, output_path: Path, sample: torch.Tensor) -> float:
    """Return the maximum absolute PyTorch/ONNX Runtime probability difference."""
    model.eval()
    with torch.no_grad():
        pytorch = model(sample).detach().cpu().numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_probability = session.run(["probability"], {"sequences": sample.detach().cpu().numpy()})[0]
    return float(np.max(np.abs(pytorch - onnx_probability)))
