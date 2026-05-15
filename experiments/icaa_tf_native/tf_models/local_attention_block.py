# ICAA17K DAT 첫 로컬 어텐션 블록을 TensorFlow/Keras로 구현한 모듈
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _tensor_from_state(state_dict, name: str):
    return state_dict[name].detach().cpu().numpy()


def _set_dense_from_linear(layer: tf.keras.layers.Dense, weight, bias) -> None:
    layer.set_weights([np.transpose(weight, (1, 0)), bias])


class TFLocalAttention(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `LocalAttention` for stage 0 block 0."""

    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        window_size: int = 7,
        name: str = "tf_local_attention_stage0_block0",
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads
        self.window_size = (window_size, window_size)
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.proj_qkv = tf.keras.layers.Dense(3 * dim, use_bias=True, name="proj_qkv")
        self.proj_out = tf.keras.layers.Dense(dim, use_bias=True, name="proj_out")
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=((2 * window_size - 1) * (2 * window_size - 1), heads),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.relative_position_index = None

    def call(self, inputs, attention_mask=None, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        shape = tf.shape(x)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        window_h, window_w = self.window_size
        rows = height // window_h
        cols = width // window_w
        tokens_per_window = window_h * window_w

        x_windows = tf.reshape(x, [batch, rows, window_h, cols, window_w, channels])
        x_windows = tf.transpose(x_windows, [0, 1, 3, 2, 4, 5])
        x_windows = tf.reshape(x_windows, [batch * rows * cols, tokens_per_window, channels])

        qkv = self.proj_qkv(x_windows)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = q * tf.cast(self.scale, q.dtype)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        attn = tf.einsum("bhmc,bhnc->bhmn", q, k)

        if self.relative_position_index is None:
            raise RuntimeError("relative_position_index must be loaded from the PyTorch state_dict before calling.")
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1]),
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [tokens_per_window, tokens_per_window, self.heads],
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if attention_mask is not None:
            n_windows = rows * cols
            mask = tf.cast(attention_mask, attn.dtype)
            attn = tf.reshape(attn, [batch, n_windows, self.heads, tokens_per_window, tokens_per_window])
            attn = attn + tf.reshape(mask, [1, n_windows, 1, tokens_per_window, tokens_per_window])
            attn = tf.reshape(attn, [batch * n_windows, self.heads, tokens_per_window, tokens_per_window])

        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.einsum("bhmn,bhnc->bhmc", attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch * rows * cols, tokens_per_window, self.dim])
        out = self.proj_out(out)

        out = tf.reshape(out, [batch, rows, cols, window_h, window_w, self.dim])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        return tf.reshape(out, [batch, height, width, self.dim])

    def _split_heads(self, value):
        shape = tf.shape(value)
        batch_windows = shape[0]
        tokens = shape[1]
        value = tf.reshape(value, [batch_windows, tokens, self.heads, self.head_dim])
        return tf.transpose(value, [0, 2, 1, 3])

    def load_from_pytorch_state_dict(self, state_dict, prefix: str) -> None:
        _set_dense_from_linear(
            self.proj_qkv,
            _tensor_from_state(state_dict, f"{prefix}.proj_qkv.weight"),
            _tensor_from_state(state_dict, f"{prefix}.proj_qkv.bias"),
        )
        _set_dense_from_linear(
            self.proj_out,
            _tensor_from_state(state_dict, f"{prefix}.proj_out.weight"),
            _tensor_from_state(state_dict, f"{prefix}.proj_out.bias"),
        )
        self.relative_position_bias_table.assign(_tensor_from_state(state_dict, f"{prefix}.relative_position_bias_table"))
        self.relative_position_index = tf.convert_to_tensor(
            _tensor_from_state(state_dict, f"{prefix}.relative_position_index"),
            dtype=tf.int32,
        )


class TFTransformerMLP(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `TransformerMLP` without dropout."""

    def __init__(
        self,
        channels: int = 128,
        expansion: int = 4,
        name: str = "tf_transformer_mlp_stage0_block0",
    ) -> None:
        super().__init__(name=name)
        self.channels = channels
        self.expansion = expansion
        self.hidden_channels = channels * expansion
        self.linear1 = tf.keras.layers.Dense(self.hidden_channels, use_bias=True, name="linear1")
        self.linear2 = tf.keras.layers.Dense(channels, use_bias=True, name="linear2")

    def call(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        shape = tf.shape(x)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        x = tf.reshape(x, [batch, height * width, channels])
        x = self.linear1(x)
        x = tf.nn.gelu(x, approximate=False)
        x = self.linear2(x)
        return tf.reshape(x, [batch, height, width, self.channels])

    def load_from_pytorch_state_dict(self, state_dict, prefix: str) -> None:
        _set_dense_from_linear(
            self.linear1,
            _tensor_from_state(state_dict, f"{prefix}.chunk.linear1.weight"),
            _tensor_from_state(state_dict, f"{prefix}.chunk.linear1.bias"),
        )
        _set_dense_from_linear(
            self.linear2,
            _tensor_from_state(state_dict, f"{prefix}.chunk.linear2.weight"),
            _tensor_from_state(state_dict, f"{prefix}.chunk.linear2.bias"),
        )


class TFFirstLocalAttentionBlock(tf.keras.layers.Layer):
    """NHWC equivalent of DAT stage 0 depth-0 local-attention block."""

    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        window_size: int = 7,
        expansion: int = 4,
        epsilon: float = 1e-5,
        name: str = "tf_first_local_attention_block",
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.expansion = expansion
        self.epsilon = epsilon
        self.norm0 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name="layer_norm_0")
        self.attn = TFLocalAttention(dim=dim, heads=heads, window_size=window_size, name="local_attention")
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name="layer_norm_1")
        self.mlp = TFTransformerMLP(channels=dim, expansion=expansion, name="mlp")

    def call(self, inputs, training: bool = False):
        output, _debug = self.call_with_debug(inputs, training=training)
        return output

    def call_with_debug(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        debug = {}

        x0 = x
        norm0 = self.norm0(x)
        debug["norm0"] = norm0
        attn_out = self.attn(norm0, training=False)
        debug["attn_out"] = attn_out
        after_attn_residual = attn_out + x0
        debug["after_attn_residual"] = after_attn_residual

        x1 = after_attn_residual
        norm1 = self.norm1(after_attn_residual)
        debug["norm1"] = norm1
        mlp_out = self.mlp(norm1, training=False)
        debug["mlp_out"] = mlp_out
        output = mlp_out + x1
        debug["output"] = output
        return output, debug

    def build_for_input_shape(self, input_shape=(1, 56, 56, 128)) -> None:
        if self.attn.relative_position_index is None:
            tokens = self.window_size * self.window_size
            self.attn.relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "stages.0") -> None:
        self.norm0.set_weights(
            [
                _tensor_from_state(state_dict, f"{prefix}.layer_norms.0.norm.weight"),
                _tensor_from_state(state_dict, f"{prefix}.layer_norms.0.norm.bias"),
            ]
        )
        self.attn.load_from_pytorch_state_dict(state_dict, f"{prefix}.attns.0")
        self.norm1.set_weights(
            [
                _tensor_from_state(state_dict, f"{prefix}.layer_norms.1.norm.weight"),
                _tensor_from_state(state_dict, f"{prefix}.layer_norms.1.norm.bias"),
            ]
        )
        self.mlp.load_from_pytorch_state_dict(state_dict, f"{prefix}.mlps.0")

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "input_layout": "NHWC/channels_last",
            "target_pytorch_block": "stages.0 depth index 0",
            "layers": [
                {
                    "name": "layer_norm_0",
                    "pytorch_class": "LayerNormProxy",
                    "tf_layer": "tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)",
                },
                {
                    "name": "local_attention",
                    "pytorch_class": "LocalAttention",
                    "heads": self.heads,
                    "window_size": [self.window_size, self.window_size],
                    "qkv_projection": "Dense mapped from proj_qkv Linear",
                    "relative_position_bias": "gather table with relative_position_index",
                },
                {
                    "name": "residual_0",
                    "operation": "attn_out + block_input",
                    "drop_path": "identity for stage 0 block 0",
                },
                {
                    "name": "layer_norm_1",
                    "pytorch_class": "LayerNormProxy",
                    "tf_layer": "tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)",
                },
                {
                    "name": "mlp",
                    "pytorch_class": "TransformerMLP",
                    "operation": "Dense -> exact GELU -> Dense, dropout omitted in eval",
                },
                {
                    "name": "residual_1",
                    "operation": "mlp_out + after_attn_residual",
                    "drop_path": "identity for stage 0 block 0",
                },
            ],
        }
