# ICAA17K DAT 유틸리티 및 수동 레이어 구현
from __future__ import annotations

import numpy as np
import tensorflow as tf

class TFManualLayerNorm(tf.keras.layers.Layer):
    """TFLite-friendly manual LayerNormalization with extreme robustness."""
    def __init__(self, epsilon: float = 1e-5, name: str = None):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(name="gamma", shape=(dim,), initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(dim,), initializer="zeros")

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        # Ensure variance is positive and handle potential precision issues
        variance = tf.maximum(variance, 0.0)
        # TFLite rsqrt can be unstable with very small values
        std = tf.math.sqrt(variance + self.epsilon)
        return (x - mean) / std * self.gamma + self.beta

def manual_gelu(x):
    """Manual TFLite-friendly GELU implementation (approximate)."""
    # Using the standard tanh approximation
    return 0.5 * x * (1.0 + tf.tanh(0.7978845608 * (x + 0.044715 * tf.pow(x, 3))))

def manual_sigmoid(x):
    """Manual TFLite-friendly Sigmoid implementation."""
    # Clip input to avoid overflow in exp or tanh
    x = tf.clip_by_value(x, -20.0, 20.0)
    return 0.5 * (tf.tanh(x * 0.5) + 1.0)
