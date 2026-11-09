# ICAA17K DAT SoftHistogram 모듈을 TensorFlow/Keras로 구현한 레이어
from __future__ import annotations

import numpy as np
import tensorflow as tf


class TFSoftHistogram(tf.keras.layers.Layer):
    """TensorFlow equivalent of export-safe PyTorch SoftHistogram."""

    def __init__(
        self,
        n_features: int,
        n_examples: int,
        num_bins: int,
        quantiles: bool = False,
        name: str = "tf_soft_histogram",
    ) -> None:
        super().__init__(name=name)
        self.in_channels = n_features
        self.n_examples = n_examples
        self.num_bins = num_bins
        self.quantiles = quantiles
        self.total_bins = n_features * num_bins

        bin_centers = -1.0 / num_bins * (np.arange(num_bins, dtype=np.float32) + 0.5)
        repeated_centers = np.tile(bin_centers, n_features)
        bin_width = -num_bins * 2.0

        self.centers = self.add_weight(
            name="centers",
            shape=(self.total_bins,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.bin_centers_conv_weight = self.add_weight(
            name="bin_centers_conv_weight",
            shape=(self.total_bins,),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
        )
        self.bin_centers_conv_bias = self.add_weight(
            name="bin_centers_conv_bias",
            shape=(self.total_bins,),
            initializer=tf.keras.initializers.Constant(repeated_centers),
            trainable=True,
        )
        self.bin_widths_conv_weight = self.add_weight(
            name="bin_widths_conv_weight",
            shape=(self.total_bins,),
            initializer=tf.keras.initializers.Constant(bin_width),
            trainable=True,
        )
        self.bin_widths_conv_bias = self.add_weight(
            name="bin_widths_conv_bias",
            shape=(self.total_bins,),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
        )
        self.widths = self.bin_widths_conv_weight

    def call(self, inputs, training: bool = False):
        del training
        inputs = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        if inputs.shape.rank != 2:
            raise ValueError(f"TFSoftHistogram expects rank-2 [batch, features] input, got shape {inputs.shape}.")

        features_by_batch = tf.transpose(inputs, perm=(1, 0))
        expanded = tf.repeat(features_by_batch, repeats=self.num_bins, axis=0)

        centers_weight = tf.reshape(self.bin_centers_conv_weight, (-1, 1))
        centers_bias = tf.reshape(self.bin_centers_conv_bias, (-1, 1))
        widths_weight = tf.reshape(self.bin_widths_conv_weight, (-1, 1))
        widths_bias = tf.reshape(self.bin_widths_conv_bias, (-1, 1))

        output = expanded * centers_weight + centers_bias
        output = tf.abs(output)
        output = output * widths_weight + widths_bias
        output = tf.nn.relu(output)
        output = tf.expand_dims(output, axis=0)

        if self.quantiles:
            output = tf.math.cumsum(tf.reshape(output, (-1, self.num_bins)), axis=1)

        return output

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "hist_feature.") -> None:
        def value(name: str):
            return state_dict[f"{prefix}{name}"].detach().cpu().numpy()

        self.centers.assign(value("centers"))
        self.bin_centers_conv_weight.assign(value("bin_centers_conv.weight").reshape(-1))
        self.bin_centers_conv_bias.assign(value("bin_centers_conv.bias"))
        self.bin_widths_conv_weight.assign(value("bin_widths_conv.weight").reshape(-1))
        self.bin_widths_conv_bias.assign(value("bin_widths_conv.bias"))

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "constructor_args": {
                "n_features": self.in_channels,
                "n_examples": self.n_examples,
                "num_bins": self.num_bins,
                "quantiles": self.quantiles,
            },
            "input_shape": "[batch, n_features]",
            "output_shape_when_quantiles_false": "[1, n_features * num_bins, batch]",
            "math": [
                "transpose input [batch, features] -> [features, batch]",
                "repeat each feature row num_bins times to emulate grouped Conv1d expansion",
                "expanded * bin_centers_conv.weight + bin_centers_conv.bias",
                "abs",
                "value * bin_widths_conv.weight + bin_widths_conv.bias",
                "relu",
            ],
            "variables": [
                {"name": "centers", "trainable": True, "used_in_forward": False},
                {"name": "bin_centers_conv_weight", "trainable": False, "used_in_forward": True},
                {"name": "bin_centers_conv_bias", "trainable": True, "used_in_forward": True},
                {"name": "bin_widths_conv_weight", "trainable": True, "used_in_forward": True},
                {"name": "bin_widths_conv_bias", "trainable": False, "used_in_forward": True},
            ],
        }

