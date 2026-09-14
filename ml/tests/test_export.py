import onnx
import torch

from ml.data_contract import FEATURE_COLUMNS
from ml.export_onnx import export_model, verify_onnx_probabilities
from ml.models import CnnGru


def test_export_has_dynamic_batch_fixed_sequence_and_matches_pytorch(tmp_path):
    torch.manual_seed(20260821)
    model = CnnGru(feature_count=len(FEATURE_COLUMNS), hidden_size=8).eval()
    sample = torch.linspace(0.0, 1.0, 10 * len(FEATURE_COLUMNS), dtype=torch.float32).reshape(1, 10, len(FEATURE_COLUMNS))
    output_path = tmp_path / "sleep_model.onnx"

    export_model(model, output_path, feature_count=len(FEATURE_COLUMNS))
    difference = verify_onnx_probabilities(model, output_path, sample)
    graph_input = onnx.load(output_path).graph.input[0]
    dimensions = graph_input.type.tensor_type.shape.dim

    assert difference < 1e-4
    assert dimensions[0].dim_param == "batch"
    assert dimensions[1].dim_value == 10
    assert dimensions[2].dim_value == len(FEATURE_COLUMNS)
