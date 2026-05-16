# ICAA17K DAT 0번부터 3번 TransformerStage를 TensorFlow/Keras로 구현한 모듈
from __future__ import annotations

import tensorflow as tf

from tf_models.deformable_attention import TFDAttentionBaseline
from tf_models.local_attention_block import TFLocalAttention, TFTransformerMLP, _tensor_from_state


class TFShiftWindowAttention(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `ShiftWindowAttention` for local-shift DAT stages."""

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

        h, w = x.shape[1], x.shape[2]
        s = self.shift_size

        # Roll: shift_size만큼 순환 이동 (PyTorch roll(shift=-s) == TF roll(shift=-s))
        if s > 0:
            s_h = s % h
            x = tf.concat([x[:, s_h:, :, :], x[:, :s_h, :, :]], axis=1)
            s_w = s % w
            x = tf.concat([x[:, :, s_w:, :], x[:, :, :s_w, :]], axis=2)

        shifted_output = self.local_attention(x, attention_mask=self.attn_mask, training=False)

        # Reverse Roll: shift_size만큼 반대 방향으로 순환 이동
        if s > 0:
            rs_h = (h - s) % h
            shifted_output = tf.concat([shifted_output[:, rs_h:, :, :], shifted_output[:, :rs_h, :, :]], axis=1)
            rs_w = (w - s) % w
            shifted_output = tf.concat([shifted_output[:, :, rs_w:, :], shifted_output[:, :, :rs_w, :]], axis=2)

        return shifted_output

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


class TFTransformerStage1(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `TransformerStage` for DAT stage 1 only."""

    def __init__(
        self,
        dim: int = 256,
        depth: int = 2,
        stage_spec=("L", "S"),
        heads: int = 8,
        window_size: int = 7,
        expansion: int = 4,
        epsilon: float = 1e-5,
        name: str = "tf_transformer_stage1",
    ) -> None:
        super().__init__(name=name)
        if depth != 2 or tuple(stage_spec) != ("L", "S"):
            raise ValueError("TFTransformerStage1 intentionally supports only DAT stage 1 with stage_spec=('L', 'S').")
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
                fmap_size=(28, 28),
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

    def build_for_input_shape(self, input_shape=(1, 28, 28, 256)) -> None:
        tokens = self.window_size * self.window_size
        n_windows = (input_shape[1] // self.window_size) * (input_shape[2] // self.window_size)
        for attn in self.attns:
            if isinstance(attn, TFLocalAttention):
                attn.relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
            else:
                attn.local_attention.relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
                attn.attn_mask = tf.zeros((n_windows, tokens, tokens), dtype=tf.float32)
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "stages.1") -> None:
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
            "target_pytorch_module": "stages.1",
            "input_layout": "NHWC/channels_last",
            "proj": "Identity; no projection weights are used for stage 1",
            "depth": self.depth,
            "stage_spec": list(self.stage_spec),
            "blocks": [
                {
                    "index": 0,
                    "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
                    "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
                    "layer_norms": ["layer_norm_0", "layer_norm_1"],
                    "drop_path_eval": "identity in eval mode",
                },
                {
                    "index": 1,
                    "attention": "TFShiftWindowAttention mapped from PyTorch ShiftWindowAttention",
                    "shift_size": (self.window_size + 1) // 2,
                    "attn_mask": "direct copy from stages.1.attns.1.attn_mask",
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


class TFTransformerStage2(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `TransformerStage` for DAT stage 2 only."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 18,
        stage_spec=tuple(["L", "D"] * 9),
        heads: int = 16,
        window_size: int = 7,
        expansion: int = 4,
        groups: int = 4,
        epsilon: float = 1e-5,
        name: str = "tf_transformer_stage2",
    ) -> None:
        super().__init__(name=name)
        if depth != 18 or tuple(stage_spec) != tuple(["L", "D"] * 9):
            raise ValueError("TFTransformerStage2 intentionally supports only DAT stage 2 with stage_spec=('L', 'D') * 9.")
        self.dim = dim
        self.depth = depth
        self.stage_spec = tuple(stage_spec)
        self.heads = heads
        self.window_size = window_size
        self.expansion = expansion
        self.groups = groups
        self.epsilon = epsilon
        self.proj = "Identity"
        self.layer_norms = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name=f"layer_norm_{idx}")
            for idx in range(2 * depth)
        ]
        self.attns = []
        head_channels = dim // heads
        for block_idx, spec in enumerate(self.stage_spec):
            if spec == "L":
                self.attns.append(
                    TFLocalAttention(
                        dim=dim,
                        heads=heads,
                        window_size=window_size,
                        name=f"local_attention_block{block_idx}",
                    )
                )
            elif spec == "D":
                self.attns.append(
                    TFDAttentionBaseline(
                        q_size=(14, 14),
                        kv_size=(14, 14),
                        n_heads=heads,
                        n_head_channels=head_channels,
                        n_groups=groups,
                        stride=1,
                        offset_range_factor=2,
                        use_pe=True,
                        dwc_pe=False,
                        no_off=False,
                        fixed_pe=False,
                        stage_idx=2,
                        epsilon=epsilon,
                        name=f"deformable_attention_block{block_idx}",
                    )
                )
            else:
                raise ValueError(f"Unsupported stage 2 attention spec: {spec}")
        self.mlps = [
            TFTransformerMLP(channels=dim, expansion=expansion, name=f"mlp_block{idx}")
            for idx in range(depth)
        ]

    def call(self, inputs, training: bool = False):
        output, _debug = self.call_with_debug(inputs, training=training)
        return output

    def call_with_debug(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        debug = {"stage_input_after_proj": x}
        for block_idx, spec in enumerate(self.stage_spec):
            x0 = x
            norm_attn = self.layer_norms[2 * block_idx](x)
            debug[f"block{block_idx}_norm_attn"] = norm_attn
            if spec == "D":
                attn_out, attn_debug = self.attns[block_idx].call_with_debug(norm_attn, training=False)
                debug[f"block{block_idx}_dattention_pos"] = attn_debug["pos"]
                debug[f"block{block_idx}_dattention_reference"] = attn_debug["reference"]
            else:
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

    def build_for_input_shape(self, input_shape=(1, 14, 14, 512)) -> None:
        tokens = self.window_size * self.window_size
        for attn in self.attns:
            if isinstance(attn, TFLocalAttention):
                attn.relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "stages.2") -> None:
        for idx, norm in enumerate(self.layer_norms):
            norm.set_weights(
                [
                    _tensor_from_state(state_dict, f"{prefix}.layer_norms.{idx}.norm.weight"),
                    _tensor_from_state(state_dict, f"{prefix}.layer_norms.{idx}.norm.bias"),
                ]
            )
        for idx, attn in enumerate(self.attns):
            attn.load_from_pytorch_state_dict(state_dict, f"{prefix}.attns.{idx}")
        for idx, mlp in enumerate(self.mlps):
            mlp.load_from_pytorch_state_dict(state_dict, f"{prefix}.mlps.{idx}")

    def structure_summary(self) -> dict:
        blocks = []
        for idx, spec in enumerate(self.stage_spec):
            if spec == "L":
                attention_summary = "TFLocalAttention mapped from PyTorch LocalAttention"
            else:
                attention_summary = "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline"
            blocks.append(
                {
                    "index": idx,
                    "attention_type": spec,
                    "attention": attention_summary,
                    "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
                    "layer_norms": [f"layer_norm_{2 * idx}", f"layer_norm_{2 * idx + 1}"],
                    "drop_path_eval": "identity in eval mode",
                }
            )
        return {
            "class_name": self.__class__.__name__,
            "target_pytorch_module": "stages.2",
            "input_layout": "NHWC/channels_last",
            "proj": "Identity; no projection weights are used for stage 2",
            "depth": self.depth,
            "stage_spec": list(self.stage_spec),
            "blocks": blocks,
            "deformable_attention_config": {
                "q_size": [14, 14],
                "kv_size": [14, 14],
                "heads": self.heads,
                "head_channels": self.dim // self.heads,
                "groups": self.groups,
                "offset_range_factor": 2,
                "use_pe": True,
                "dwc_pe": False,
                "fixed_pe": False,
                "no_off": False,
            },
            "residual_order": [
                "x0 = x",
                "attn_out = attns[d](layer_norms[2*d](x))",
                "x = drop_path[d](attn_out) + x0",
                "x0 = x",
                "mlp_out = mlps[d](layer_norms[2*d+1](x))",
                "x = drop_path[d](mlp_out) + x0",
            ],
        }


class TFTransformerStage3(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch `TransformerStage` for DAT stage 3 only."""

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 2,
        stage_spec=("L", "D"),
        heads: int = 32,
        window_size: int = 7,
        expansion: int = 4,
        groups: int = 8,
        epsilon: float = 1e-5,
        name: str = "tf_transformer_stage3",
    ) -> None:
        super().__init__(name=name)
        if depth != 2 or tuple(stage_spec) != ("L", "D"):
            raise ValueError("TFTransformerStage3 intentionally supports only DAT stage 3 with stage_spec=('L', 'D').")
        self.dim = dim
        self.depth = depth
        self.stage_spec = tuple(stage_spec)
        self.heads = heads
        self.window_size = window_size
        self.expansion = expansion
        self.groups = groups
        self.epsilon = epsilon
        self.proj = "Identity"
        self.layer_norms = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name=f"layer_norm_{idx}")
            for idx in range(2 * depth)
        ]
        head_channels = dim // heads
        self.attns = [
            TFLocalAttention(
                dim=dim,
                heads=heads,
                window_size=window_size,
                name="local_attention_block0",
            ),
            TFDAttentionBaseline(
                q_size=(7, 7),
                kv_size=(7, 7),
                n_heads=heads,
                n_head_channels=head_channels,
                n_groups=groups,
                stride=1,
                offset_range_factor=2,
                use_pe=True,
                dwc_pe=False,
                no_off=False,
                fixed_pe=False,
                stage_idx=3,
                epsilon=epsilon,
                name="deformable_attention_block1",
            ),
        ]
        self.mlps = [
            TFTransformerMLP(channels=dim, expansion=expansion, name=f"mlp_block{idx}")
            for idx in range(depth)
        ]

    def call(self, inputs, training: bool = False):
        output, _debug = self.call_with_debug(inputs, training=training)
        return output

    def call_with_debug(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        debug = {"stage_input_after_proj": x}
        for block_idx, spec in enumerate(self.stage_spec):
            x0 = x
            norm_attn = self.layer_norms[2 * block_idx](x)
            debug[f"block{block_idx}_norm_attn"] = norm_attn
            if spec == "D":
                attn_out, attn_debug = self.attns[block_idx].call_with_debug(norm_attn, training=False)
                debug[f"block{block_idx}_dattention_pos"] = attn_debug["pos"]
                debug[f"block{block_idx}_dattention_reference"] = attn_debug["reference"]
            else:
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

    def build_for_input_shape(self, input_shape=(1, 7, 7, 1024)) -> None:
        tokens = self.window_size * self.window_size
        self.attns[0].relative_position_index = tf.zeros((tokens, tokens), dtype=tf.int32)
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "stages.3") -> None:
        for idx, norm in enumerate(self.layer_norms):
            norm.set_weights(
                [
                    _tensor_from_state(state_dict, f"{prefix}.layer_norms.{idx}.norm.weight"),
                    _tensor_from_state(state_dict, f"{prefix}.layer_norms.{idx}.norm.bias"),
                ]
            )
        for idx, attn in enumerate(self.attns):
            attn.load_from_pytorch_state_dict(state_dict, f"{prefix}.attns.{idx}")
        for idx, mlp in enumerate(self.mlps):
            mlp.load_from_pytorch_state_dict(state_dict, f"{prefix}.mlps.{idx}")

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "target_pytorch_module": "stages.3",
            "input_layout": "NHWC/channels_last",
            "proj": "Identity; no projection weights are used for stage 3",
            "depth": self.depth,
            "stage_spec": list(self.stage_spec),
            "blocks": [
                {
                    "index": 0,
                    "attention_type": "L",
                    "attention": "TFLocalAttention mapped from PyTorch LocalAttention",
                    "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
                    "layer_norms": ["layer_norm_0", "layer_norm_1"],
                    "drop_path_eval": "identity in eval mode",
                },
                {
                    "index": 1,
                    "attention_type": "D",
                    "attention": "TFDAttentionBaseline mapped from PyTorch DAttentionBaseline",
                    "mlp": "TFTransformerMLP mapped from PyTorch TransformerMLP",
                    "layer_norms": ["layer_norm_2", "layer_norm_3"],
                    "drop_path_eval": "identity in eval mode",
                },
            ],
            "deformable_attention_config": {
                "q_size": [7, 7],
                "kv_size": [7, 7],
                "heads": self.heads,
                "head_channels": self.dim // self.heads,
                "groups": self.groups,
                "offset_range_factor": 2,
                "use_pe": True,
                "dwc_pe": False,
                "fixed_pe": False,
                "no_off": False,
            },
            "residual_order": [
                "x0 = x",
                "attn_out = attns[d](layer_norms[2*d](x))",
                "x = drop_path[d](attn_out) + x0",
                "x0 = x",
                "mlp_out = mlps[d](layer_norms[2*d+1](x))",
                "x = drop_path[d](mlp_out) + x0",
            ],
        }
