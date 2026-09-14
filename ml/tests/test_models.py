import pytest
import torch

from ml.data_contract import FEATURE_COLUMNS
from ml.models import CnnGru, CnnLstm, RawCnnRecurrent, neural_candidates


def test_cnn_gru_returns_one_probability_for_each_input_sequence():
    model = CnnGru(feature_count=len(FEATURE_COLUMNS), hidden_size=8)

    probabilities = model(torch.zeros(4, 10, len(FEATURE_COLUMNS)))

    assert probabilities.shape == (4, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_cnn_lstm_uses_only_past_epochs():
    model = CnnLstm(feature_count=len(FEATURE_COLUMNS), hidden_size=8)

    assert model.lstm.bidirectional is False
    assert model.convolution.padding[0] == 0


def test_candidate_factory_builds_each_approved_hidden_size_for_both_recurrent_types():
    candidates = neural_candidates(feature_count=len(FEATURE_COLUMNS), hidden_sizes=(16, 32, 64))

    assert list(candidates) == ["cnn_gru_16", "cnn_gru_32", "cnn_gru_64", "cnn_lstm_16", "cnn_lstm_32", "cnn_lstm_64"]
    assert candidates["cnn_gru_64"].gru.hidden_size == 64
    assert candidates["cnn_lstm_16"].lstm.hidden_size == 16


@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
def test_raw_cnn_recurrent_returns_one_probability_per_raw_window(recurrent):
    model = RawCnnRecurrent(samples_per_epoch=8, channels=5, hidden_size=4, recurrent=recurrent)

    probabilities = model(torch.zeros(3, 6, 8, 5))

    assert probabilities.shape == (3, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert model.recurrent.bidirectional is False


def test_raw_cnn_recurrent_rejects_inputs_that_do_not_match_its_raw_window_contract():
    model = RawCnnRecurrent(samples_per_epoch=8, channels=5)

    with pytest.raises(ValueError, match=r"\[batch, epochs, samples, channels\]"):
        model(torch.zeros(3, 6, 8))
    with pytest.raises(ValueError, match="samples_per_epoch"):
        model(torch.zeros(3, 6, 7, 5))
    with pytest.raises(ValueError, match="channels"):
        model(torch.zeros(3, 6, 8, 4))


def test_raw_cnn_recurrent_rejects_unknown_recurrent_type():
    with pytest.raises(ValueError, match="recurrent"):
        RawCnnRecurrent(samples_per_epoch=8, channels=5, recurrent="rnn")


@pytest.mark.parametrize("model_type", [CnnGru, CnnLstm])
def test_sequence_models_reject_feature_widths_that_do_not_match_their_contract(model_type):
    model = model_type(feature_count=5, hidden_size=4)

    with pytest.raises(ValueError, match="feature_count"):
        model(torch.zeros(2, 4, 4))


@pytest.mark.parametrize("model_type", [CnnGru, CnnLstm])
def test_sequence_models_reject_epoch_sequences_shorter_than_the_convolution_kernel(model_type):
    model = model_type(feature_count=5, hidden_size=4)

    with pytest.raises(ValueError, match="at least 3"):
        model(torch.zeros(2, 2, 5))


def test_raw_cnn_recurrent_rejects_windows_with_no_epochs():
    model = RawCnnRecurrent(samples_per_epoch=8, channels=5)

    with pytest.raises(ValueError, match="at least one epoch"):
        model(torch.zeros(2, 0, 8, 5))


@pytest.mark.parametrize(
    ("samples_per_epoch", "channels", "expected_message"),
    [(2, 5, "samples_per_epoch"), (8, 0, "channels")],
)
def test_raw_cnn_recurrent_rejects_invalid_raw_input_dimensions(samples_per_epoch, channels, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        RawCnnRecurrent(samples_per_epoch=samples_per_epoch, channels=channels)


def test_raw_cnn_recurrent_prediction_for_a_causal_prefix_excludes_appended_future_epochs():
    model = RawCnnRecurrent(samples_per_epoch=8, channels=5, hidden_size=4).eval()
    prefix = torch.arange(4 * 8 * 5, dtype=torch.float32).reshape(1, 4, 8, 5)
    first_future = torch.full((1, 2, 8, 5), 100.0)
    second_future = torch.full((1, 2, 8, 5), -100.0)

    with torch.no_grad():
        prefix_probability = model(prefix)
        first_causal_probability = model(torch.cat((prefix, first_future), dim=1)[:, :4])
        second_causal_probability = model(torch.cat((prefix, second_future), dim=1)[:, :4])

    torch.testing.assert_close(prefix_probability, first_causal_probability)
    torch.testing.assert_close(prefix_probability, second_causal_probability)
