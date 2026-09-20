"""CNN-BiLSTM hybrid baseline."""

from tensorflow.keras import layers, models

from .common import compile_model


def get_cnn_bilstm_params():
    """CNN-BiLSTM hybrid baseline adapted from Hnamte and Hussain (2023),
    Telematics and Informatics Reports (10.1016/j.teler.2023.100053),
    'DCNNBiLSTM: an efficient hybrid deep learning-based intrusion
    detection system'.

    Notably, this source paper evaluates on CICIDS2018 and Edge-IIoT,
    the same dataset families used elsewhere in this study.

    Confirmed directly from the paper's Methodology section: ReLU
    activation throughout the network, Adam optimizer, learning rate
    0.0001, and categorical cross-entropy loss (one-hot encoded labels) —
    this loss choice already matches the convention used throughout this
    pipeline.

    The paper does not report per-layer filter counts, kernel size,
    BiLSTM units, dense units, dropout rate, batch size, or epoch count —
    these are not stated anywhere in the published article (its
    accompanying code repository requires contacting the corresponding
    author and was not accessible for this work). The values below for
    those specific settings are pipeline-consistent defaults, not sourced
    from the paper.
    """
    return {
        "n_conv": 1,
        "filters": 64,            # not stated in paper; pipeline default
        "kernel_size": 3,         # not stated in paper; pipeline default
        "lstm_units": 64,         # not stated in paper; pipeline default
        "dense_units": 64,        # not stated in paper; pipeline default
        "dropout": 0.3,           # not stated in paper; pipeline default
        "learning_rate": 0.0001,  # sourced: "0.0001 learning rate settings"
        "batch_size": 64,         # not stated in paper; pipeline default
        "epochs": 20,             # not stated in paper; pipeline default
    }


def build_cnn_bilstm_model(input_shape, n_classes, params=None):
    params = params or get_cnn_bilstm_params()
    model = models.Sequential(name="CNN_BiLSTM")
    model.add(layers.Input(shape=input_shape))

    for i in range(params["n_conv"]):
        model.add(
            layers.Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                activation="relu",
                padding="same",
                name=f"cnn_bilstm_conv1d_{i + 1}",
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(params["dropout"]))

    model.add(
        layers.Bidirectional(
            layers.LSTM(params["lstm_units"], return_sequences=False)
        )
    )
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return compile_model(model, params["learning_rate"])
