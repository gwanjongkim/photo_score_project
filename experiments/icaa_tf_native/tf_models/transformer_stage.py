# ICAA17K DAT 0번 TransformerStage를 TensorFlow/Keras로 구현한 모듈
from __future__ import annotations

import tensorflow as tf

from tf_models.local_attention_block import TFLocalAttention, TFTransformerMLP, _tensor_from_state


class TFShiftWindowAttention(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `ShiftWindowAttention` for DAT stage 0."""

    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        window_size: int = 7,
        shift_size: int = 4,
        fmap_size=(56, 56),
        name: str = "tf_shift_window_attention_stage0_block1",
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.fmap_size = tuple(fmap_size)
        self.local_attention = TFLocalAttention(
            dim=dim,
            heads=heads,
            window_size=window_size,
            name="shifted_local_attention",
        )
        self.attn_mask = None

    def call(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        if self.attn_mask is None:
            raise RuntimeError("attn_mask must be loaded from the PyTorch state_dict before calling.")
        shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        shifted_output = self.local_attention(shifted_x, attention_mask=self.attn_mask, training=False)
        return tf.roll(shifted_output, shift=[self.shift_size, self.shift_size], axis=[1, 2])

    def load_from_pytorch_state_dict(self, state_dict, prefix: str) -> None:
        self.local_attention.load_from_pytorch_state_dict(state_dict, prefix)
        self.attn_mask = tf.convert_to_tensor(_tensor_from_state(state_dict, f"{prefix}.attn_mask"), dtype=tf.float32)


class TFTransformerStage0(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `TransformerStage` for DAT stage 0 only."""

    def __init__(
        self,
        dim: int = 128,
        depth: int = 2,
        stage_spec=("L", "S"),
        heads: int = 4,
        window_size: int = 7,
        expansion: int = 4,
        epsilon: float = 1e-5,
        name: str = "tf_transformer_stage0",
    ) -> None:
        super().__init__(name=name)
        if depth != 2 or tuple(stage_spec) != ("L", "S"):
            raise ValueError("TFTransformerStage0 intentionally supports only DAT stage 0 with stage_spec=('L', 'S').")
        self.dim = dim
        self.depth = depth
        self.stage_spec = tuple(stage_spec)
        self.heads = heads
        self.window_size = window_size
        self.expansion = expansion
        self.epsilon = epsilon
        self.proj = "Identity"
        self.layer_norms = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name=f"layer_norm_{idx}")
            for idx in range(2 * depth)
        ]
        self.attns = [
            TFLocalAttention(dim=dim, heads=heads, window_size=window_size, name="local_attention_block0"),
            TFShiftWindowAttention(
                dim=dim,
                heads=heads,
                window_size=window_size,
                shift_size=(window_size + 1) // 2,
                fmap_size=(56, 56),
                name="shift_window_attention_block1",
            ),
        ]
        self.mlps = [
            TFTransformerMLP(channels=dim, expansion=expansion, name="mlp_block0"),
            TFTransformerMLP(channels=dim, expansion=expansion, name="mlp_block1"),
        ]

    def call(self, inputs, training: bool = False):
        output, _debug = self.call_with_debug(inputs, training=training)
        return output

    def call_with_debug(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        debug = {"stage_input_after_proj": x}
        for block_idx in range(self.depth):
            x0 = x
            norm_attn = self.layer_norms[2 * block_idx](x)
            debug[f"block{block_idx}_norm_attn"] = norm_attn
            attn_out = self.attns[block_idx](norm_attn, training=False)
            debug[f"block{block_idx}_attn_out"] = attn_out
            x = attn_out + x0
            debug[f"block{block_idx}_after_attn_residual"] = x

            x0 = x
            norm_mlp = self.layer_norms[2 * block_idx + 1](x)
            debug[f"block{block_idx}_norm_mlp"] = norm_mlp
            mlp_out = self.mlps[block_idx](norm_mlp, training=False)
            debug[f"block{block_idx}_mlp_out"] = mlp_out
            x = mlp_out + x0
            debug[f"block{block_idx}_output"] = x
        debug["stage_output"] = x
        return x, debug

    def build_for_input_shape(self, input_shape=(1, 56, 56, 128)) -> None:
        tokens = self.window_size * self.window_size
        n_windows = (input_shape[1] // self.window_size) * (input_shape[2] // self.window_size)
        for attn in self.attns:
            if isinstance(attn, TFLocalAttention):
                attn.relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
            else:
                attn.local_attention.relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
                attn.attn_mask = tf.zeros((n_windows, tokens, tokens), dtype=tf.float32)
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "stages.0") -> None:
        for idx, norm in enumerate(self.layer_norms):
            norm.set_weights(
                [
                    _tensor_from_state(state_dict, f"{prefix}.layer_norms.{idx}.norm.weight"),
                    _tensor_from_state(state_dict, f"{prefix}.layer_norms.{idx}.norm.bias"),
                ]
            )
        self.attns[0].load_from_pytorch_state_dict(state_dict, f"{prefix}.attns.0")
        self.attns[1].load_from_pytorch_state_dict(state_dict, f"{prefix}.attns.1")
        for idx, mlp in enumerate(self.mlps):
            mlp.load_from_pytorch_state_dict(state_dict, f"{prefix}.mlps.{idx}")

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "target_pytorch_module": "stages.0",
            "input_layout": "NHWC/channels_last",
            "proj": "Identity; no projection weights are used for stage 0",
            "depth": self.depth,
            "stage_spec": list(self.stage_spec),
            "blocks": [
                {
                    "index": 0,
                    "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
                    "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
                    "layer_norms": ["layer_norm_0", "layer_norm_1"],
                    "drop_path_eval": "identity",
                },
                {
                    "index": 1,
                    "attention": "TFShiftWindowAttention mapped from PyTorch ShiftWindowAttention",
                    "shift_size": (self.window_size + 1) // 2,
                    "attn_mask": "direct copy from stages.0.attns.1.attn_mask",
                    "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
                    "layer_norms": ["layer_norm_2", "layer_norm_3"],
                    "drop_path_eval": "identity in eval mode",
                },
            ],
            "residual_order": [
                "x0 = x",
                "attn_out = attns[d](layer_norms[2*d](x))",
                "x = drop_path[d](attn_out) + x0",
                "x0 = x",
                "mlp_out = mlps[d](layer_norms[2*d+1](x))",
                "x = drop_path[d](mlp_out) + x0",
            ],
        }
