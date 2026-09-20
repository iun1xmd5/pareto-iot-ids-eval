"""CNN-GRU hybrid baseline.

All baseline architectures in this package operate on the SAME sliding-window
sequence input as the proposed CNN-LSTM (shape = (batch, window, n_features)),
so they are directly comparable to it.
"""

from tensorflow.keras import layers, models

from .common import compile_model


def get_cnn_gru_params():
    """CNN-GRU hybrid baseline adapted from Imrana et al. (2024), Complex &
    Intelligent Systems (10.1007/s40747-023-01313-y), 'CNN-GRU-FF: a
    double-layer feature fusion-based network intrusion detection system'.

    All values below are taken directly from the paper's Section
    'Environment settings and hyper-parameter tuning': 64 convolution
    filters for the CNN component, 75 neurons for the GRU hidden layer,
    128 neural units for the feature-fusion MLP hidden layer, Adam
    optimizer with learning rate 8x10^-3, batch size 500, ReLU activation
    throughout (softmax at output), and a dropout of 0.5 added to the
    pooling layers. The paper trains for 100 epochs (K=10 fold
    cross-validation, reported in the Results section).

    One documented difference from this pipeline's other models: the
    source paper uses a modified focal loss (not categorical
    cross-entropy) to handle class imbalance. This pipeline uses
    categorical cross-entropy for this baseline as for all other models,
    for a fair, consistent comparison across the whole study; this is a
    deliberate deviation from the source paper, not an oversight.
    """
    return {
        "n_conv": 1,
        "filters": 64,           # sourced: "number of convolution kernels to 64"
        "kernel_size": 3,        # not stated in paper; pipeline default
        "gru_units": 75,         # sourced: "75 neurons for the hidden layers" (GRU)
        "dense_units": 128,      # sourced: "hidden layers of the MLP ... 128 neural units"
        "dropout": 0.5,          # sourced: "dropout of 0.5 to the pooling layers"
        "learning_rate": 0.008,  # sourced: "learning rate of 8x10^-3"
        "batch_size": 500,       # sourced: "batch_size of 500"
        "epochs": 100,           # sourced: "algorithm runs for 100 epochs"
    }


def build_cnn_gru_model(input_shape, n_classes, params=None):
    params = params or get_cnn_gru_params()
    model = models.Sequential(name="CNN_GRU")
    model.add(layers.Input(shape=input_shape))

    for i in range(params["n_conv"]):
        model.add(
            layers.Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                activation="relu",
                padding="same",
                name=f"cnn_gru_conv1d_{i + 1}",
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(params["dropout"]))

    model.add(layers.GRU(params["gru_units"], return_sequences=False))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return compile_model(model, params["learning_rate"])
