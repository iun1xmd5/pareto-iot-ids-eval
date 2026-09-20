"""LSTM-only ablation model (no convolutional front end)."""

from tensorflow.keras import layers, models

from .common import compile_model


def get_lstm_only_params():
    """LSTM-only ablation setting adapted from Ogunseyi and Thiyagarajan (2025).

    Their LSTM IDS setting reports one LSTM layer with 64 units, dropout 0.2,
    learning rate 0.001, batch size 64, 20 epochs, and Adam optimizer.
    """
    return {
        "lstm_units": 64,
        "dropout": 0.2,
        "dense_units": 64,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 20,
    }


def build_lstm_only_model(input_shape, n_classes, params=None):
    params = params or get_lstm_only_params()
    model = models.Sequential(name="LSTM_ONLY")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(params["lstm_units"], return_sequences=False))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return compile_model(model, params["learning_rate"])
