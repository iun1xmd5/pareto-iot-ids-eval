"""CNN-only ablation model (no recurrent layer)."""

from tensorflow.keras import layers, models

from .common import compile_model


def get_cnn_only_params():
    """CNN-only ablation setting adapted from CNN-based IDS literature.

    The cited CNN IDS paper by Kim et al. uses a CNN design for network intrusion
    detection, and survey summaries describe it as a three-convolution/two-pooling
    CNN configuration. This notebook uses a 1D adaptation because the present
    datasets are selected flow-feature vectors, not images.
    """
    return {
        "conv_filters": [64, 128, 128],
        "kernel_size": 3,
        "pool_size": 2,
        "n_pool": 2,
        "dropout": 0.3,
        "dense_units": 64,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 20,
    }


def build_cnn_only_model(input_shape, n_classes, params=None):
    params = params or get_cnn_only_params()
    conv_filters = params.get("conv_filters") or [params.get("filters", 64)] * params.get("n_conv", 2)
    n_pool = params.get("n_pool", min(2, len(conv_filters)))

    model = models.Sequential(name="CNN_ONLY")
    model.add(layers.Input(shape=input_shape))

    for i, filters in enumerate(conv_filters):
        model.add(
            layers.Conv1D(
                filters=filters,
                kernel_size=params["kernel_size"],
                activation="relu",
                padding="same",
                name=f"cnn_only_conv1d_{i + 1}",
            )
        )
        model.add(layers.BatchNormalization())
        if i < n_pool:
            model.add(layers.MaxPooling1D(pool_size=params.get("pool_size", 2)))
        model.add(layers.Dropout(params["dropout"]))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return compile_model(model, params["learning_rate"])
