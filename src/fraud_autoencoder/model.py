"""Autoencoder architecture and reconstruction-error scoring."""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def build_autoencoder(
    input_dimension: int,
    latent_dimension: int = 8,
    hidden_dimension: int = 32,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    if input_dimension < 2 or latent_dimension < 1 or hidden_dimension <= latent_dimension:
        raise ValueError("Autoencoder dimensions must satisfy input >= 2 and hidden > latent >= 1.")
    inputs = tf.keras.Input(shape=(input_dimension,), name="scaled_transaction")
    x = tf.keras.layers.Dense(hidden_dimension, activation="relu", name="encoder_hidden")(inputs)
    encoded = tf.keras.layers.Dense(latent_dimension, activation="relu", name="bottleneck")(x)
    x = tf.keras.layers.Dense(hidden_dimension, activation="relu", name="decoder_hidden")(encoded)
    outputs = tf.keras.layers.Dense(input_dimension, activation="linear", name="reconstruction")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="normal_transaction_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def reconstruction_errors(model: tf.keras.Model, scaled_features: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(scaled_features, batch_size=2048, verbose=0)
    return np.mean(np.square(scaled_features - reconstructed), axis=1).astype(np.float64)


def per_feature_errors(model: tf.keras.Model, scaled_features: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(scaled_features, batch_size=2048, verbose=0)
    return np.square(scaled_features - reconstructed).astype(np.float64)
