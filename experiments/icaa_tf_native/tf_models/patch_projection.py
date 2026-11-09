# ICAA17K DAT 패치 투영 블록을 TensorFlow/Keras로 구현한 모듈
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _tensor_from_state(state_dict, name: str):
    return state_dict[name].detach().cpu().numpy()


class TFPatchProjection(tf.keras.layers.Layer):
    """NHWC TensorFlow equivalent of the PyTorch `PatchEmbed` using Conv2d."""

    def __init__(
        self,
        dim_stem: int = 128,
        patch_size: int = 4,
        epsilon: float = 1e-5,
        name: str = "tf_patch_proj",
    ) -> None:
        super().__init__(name=name)
        self.dim_stem = dim_stem
        self.patch_size = patch_size
        self.epsilon = epsilon
        self.conv = tf.keras.layers.Conv2D(
            filters=dim_stem,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="valid",
            use_bias=True,
            name="patch_proj_conv",
        )
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name="patch_proj_norm")

    def call(self, inputs, training: bool = False):
        del training
        x = self.conv(inputs)
        return self.norm(x)

    def build_for_input_shape(self, input_shape=(1, 224, 224, 3)) -> None:
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "patch_proj") -> None:
        conv_weight = _tensor_from_state(state_dict, f"{prefix}.0.weight")
        # PyTorch [out, in, k, k] -> TF [k, k, in, out]
        conv_kernel = np.transpose(conv_weight, (2, 3, 1, 0))
        conv_bias = _tensor_from_state(state_dict, f"{prefix}.0.bias")
        self.conv.set_weights([conv_kernel, conv_bias])

        self.norm.set_weights(
            [
                _tensor_from_state(state_dict, f"{prefix}.1.norm.weight"),
                _tensor_from_state(state_dict, f"{prefix}.1.norm.bias"),
            ]
        )
