"""Causal neural candidates for 30-second sleep-feature sequences."""

import torch
from torch import nn


class _CausalCnnRecurrent(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int, recurrent: type[nn.RNNBase]) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.convolution = nn.Conv1d(feature_count, hidden_size, kernel_size=3, padding=0)
        self.recurrent = recurrent(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape [batch, sequence_epochs, feature_count]")
        if sequences.shape[2] != self.feature_count:
            raise ValueError("sequences feature dimension must match feature_count")
        if sequences.shape[1] < self.convolution.kernel_size[0]:
            raise ValueError("sequences must contain at least 3 sequence epochs")
        convolved = torch.relu(self.convolution(sequences.transpose(1, 2))).transpose(1, 2)
        encoded, _ = self.recurrent(convolved)
        return torch.sigmoid(self.classifier(encoded[:, -1, :]))


class CnnGru(_CausalCnnRecurrent):
    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__(feature_count, hidden_size, nn.GRU)
        self.gru = self.recurrent


class CnnLstm(_CausalCnnRecurrent):
    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__(feature_count, hidden_size, nn.LSTM)
        self.lstm = self.recurrent


class RawCnnRecurrent(nn.Module):
    """Encode causal raw samples within each epoch before recurrent aggregation."""

    def __init__(
        self,
        samples_per_epoch: int,
        channels: int,
        hidden_size: int = 32,
        recurrent: str = "gru",
    ) -> None:
        super().__init__()
        if recurrent not in {"gru", "lstm"}:
            raise ValueError("recurrent must be either 'gru' or 'lstm'")
        if samples_per_epoch < 3:
            raise ValueError("samples_per_epoch must be at least 3")
        if channels < 1:
            raise ValueError("channels must be positive")

        self.samples_per_epoch = samples_per_epoch
        self.channels = channels
        self.convolution = nn.Conv1d(channels, hidden_size, kernel_size=3, padding=0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        recurrent_type = nn.GRU if recurrent == "gru" else nn.LSTM
        self.recurrent = recurrent_type(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4:
            raise ValueError("windows must have shape [batch, epochs, samples, channels]")
        if windows.shape[2] != self.samples_per_epoch:
            raise ValueError("windows samples dimension must match samples_per_epoch")
        if windows.shape[3] != self.channels:
            raise ValueError("windows channels dimension must match channels")
        if windows.shape[1] < 1:
            raise ValueError("windows must contain at least one epoch")

        batch_size, epochs, _, _ = windows.shape
        raw_epochs = windows.reshape(batch_size * epochs, self.samples_per_epoch, self.channels).transpose(1, 2)
        embeddings = self.pool(torch.relu(self.convolution(raw_epochs))).squeeze(-1)
        encoded, _ = self.recurrent(embeddings.reshape(batch_size, epochs, -1))
        return torch.sigmoid(self.classifier(encoded[:, -1, :]))


class RawCnnGru(RawCnnRecurrent):
    def __init__(self, samples_per_epoch: int, channels: int, hidden_size: int = 32) -> None:
        super().__init__(samples_per_epoch, channels, hidden_size, recurrent="gru")
        self.gru = self.recurrent


class RawCnnLstm(RawCnnRecurrent):
    def __init__(self, samples_per_epoch: int, channels: int, hidden_size: int = 32) -> None:
        super().__init__(samples_per_epoch, channels, hidden_size, recurrent="lstm")
        self.lstm = self.recurrent


def neural_candidates(feature_count: int, hidden_sizes: tuple[int, ...] = (16, 32, 64)) -> dict[str, nn.Module]:
    """Create the approved causal GRU/LSTM capacity comparison set."""
    return {
        **{f"cnn_gru_{size}": CnnGru(feature_count, size) for size in hidden_sizes},
        **{f"cnn_lstm_{size}": CnnLstm(feature_count, size) for size in hidden_sizes},
    }
