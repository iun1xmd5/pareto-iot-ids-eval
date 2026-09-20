"""Shared helpers used by every model-building module in this package."""

from tensorflow.keras import optimizers


def compile_model(model, learning_rate):
    """Compile a Keras model with the Adam optimizer and sparse categorical
    cross-entropy loss, consistent across every architecture in this study.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
