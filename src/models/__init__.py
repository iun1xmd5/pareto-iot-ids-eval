"""Model architectures for the IDS study.

Every builder shares the same call signature:

    build_<name>_model(input_shape, n_classes, params=None)

and consumes the same sliding-window sequence input, shape
(batch, window, n_features), so results across models are directly
comparable. Each module also exposes a get_<name>_params() function
returning that architecture's default/paper-sourced hyperparameters.
"""

from .common import compile_model
from .cnn_lstm import build_cnn_lstm_model, get_default_cnn_lstm_params
from .cnn_only import build_cnn_only_model, get_cnn_only_params
from .lstm_only import build_lstm_only_model, get_lstm_only_params
from .cnn_gru import build_cnn_gru_model, get_cnn_gru_params
from .cnn_bilstm import build_cnn_bilstm_model, get_cnn_bilstm_params
from .transformer import build_transformer_model, get_transformer_params

__all__ = [
    "compile_model",
    "build_cnn_lstm_model",
    "get_default_cnn_lstm_params",
    "build_cnn_only_model",
    "get_cnn_only_params",
    "build_lstm_only_model",
    "get_lstm_only_params",
    "build_cnn_gru_model",
    "get_cnn_gru_params",
    "build_cnn_bilstm_model",
    "get_cnn_bilstm_params",
    "build_transformer_model",
    "get_transformer_params",
]
