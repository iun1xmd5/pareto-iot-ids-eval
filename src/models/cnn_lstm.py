"""Proposed CNN-LSTM hybrid model.

Sliding-window sequence input -> Conv1D block(s) -> LSTM -> Dense head.
"""

from tensorflow.keras import layers, models

from .common import compile_model


def get_default_cnn_lstm_params():
    """Non-optimized CNN-LSTM counterfactual baseline.

    This is a non-tuned CNN+LSTM control model adapted from hybrid CNN+LSTM IDS
    literature. The epoch and batch-size settings follow the reported CNN+LSTM
    experimental setting from Altunay and Albayrak, while the input and output
    layers are adapted to the present ToN-IoT and InSDN feature tensors.
    Early stopping is still used during training to avoid unnecessary overfitting.
    """
    return {
        "n_conv": 1,
        "filters": 64,
        "kernel_size": 3,
        "lstm_units": 64,
        "dense_units": 64,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 120,
        "epochs": 100,
    }


def build_cnn_lstm_model(input_shape, n_classes, params):
    model = models.Sequential(name="CNN_LSTM")
    model.add(layers.Input(shape=input_shape))

    for i in range(params["n_conv"]):
        model.add(
            layers.Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                activation="relu",
                padding="same",
                name=f"conv1d_{i + 1}",
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(params["dropout"]))

    model.add(layers.LSTM(params["lstm_units"], return_sequences=False))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))

    return compile_model(model, params["learning_rate"])
