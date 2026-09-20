"""Transformer encoder-only baseline."""

from tensorflow.keras import layers, models

from .common import compile_model


def get_transformer_params():
    """Transformer encoder-only IDS baseline adapted from Ataa, M.S., Sanad,
    E.E., and El-khoribi, R.A. (2024), "Intrusion detection in software
    defined network using deep learning approaches," Scientific Reports,
    14, 29159 (10.1038/s41598-024-79001-1).

    Confirmed directly from the paper's architecture description: a
    multi-headed self-attention layer with TWO heads, followed by a
    normalization layer, then FIVE Conv1D layers (each followed by
    normalization), then global average pooling, then a softmax dense
    layer. (num_heads=2, n_conv_layers=5 are taken directly from the text.)

    The paper does not report per-layer filter counts, dropout rate,
    learning rate, batch size, or epoch count — these are not stated
    anywhere in the published article. The values below for those specific
    settings are pipeline-consistent defaults, not sourced from the paper.
    """
    return {
        "num_heads": 2,          # sourced: paper states "two heads"
        "key_dim": 32,           # not stated in paper; pipeline default
        "conv_filters": [64, 64, 128, 128, 128],  # 5 layers sourced; filter counts not stated, pipeline default
        "kernel_size": 3,        # not stated in paper; pipeline default
        "dropout": 0.2,          # not stated in paper; pipeline default
        "dense_units": 64,       # not stated in paper; pipeline default
        "learning_rate": 1e-3,   # not stated in paper; pipeline default
        "batch_size": 64,        # not stated in paper; pipeline default
        "epochs": 20,            # not stated in paper; pipeline default
    }


def build_transformer_model(input_shape, n_classes, params=None):
    params = params or get_transformer_params()
    inputs = layers.Input(shape=input_shape)

    attn_output = layers.MultiHeadAttention(
        num_heads=params["num_heads"], key_dim=params["key_dim"],
        name="self_attention",
    )(inputs, inputs)
    x = layers.LayerNormalization(name="attn_norm")(attn_output)

    for i, filters in enumerate(params["conv_filters"]):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=params["kernel_size"],
            activation="relu",
            padding="same",
            name=f"transformer_conv1d_{i + 1}",
        )(x)
        x = layers.LayerNormalization(name=f"transformer_norm_{i + 1}")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(params["dropout"])(x)
    x = layers.Dense(params["dense_units"], activation="relu")(x)
    x = layers.Dropout(params["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Transformer_Encoder")
    return compile_model(model, params["learning_rate"])
