# ICAA17K DAT deformable attention 모듈을 TensorFlow/Keras로 구현한 레이어
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _tensor_from_state(state_dict, name: str):
    return state_dict[name].detach().cpu().numpy()


class PytorchLikeConv1x1(tf.keras.layers.Layer):
    """PyTorch's nn.Conv2d(kernel_size=1) equivalent in NHWC."""

    def __init__(self, out_channels: int, use_bias: bool = True, name: str = None) -> None:
        super().__init__(name=name)
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.out_channels,),
                initializer="zeros",
                trainable=True,
            )

    def call(self, x):
        shape = tf.shape(x)
        flat = tf.reshape(x, [-1, shape[-1]])
        output = tf.linalg.matmul(flat, tf.cast(self.kernel, flat.dtype))
        if self.bias is not None:
            output = output + tf.cast(self.bias, output.dtype)
        return tf.reshape(output, tf.concat([shape[:-1], [self.out_channels]], axis=0))

    def set_from_pytorch(self, weight, bias=None) -> None:
        kernel = weight[:, :, 0, 0]
        # PyTorch weight: [out, in, 1, 1] -> [in, out]
        kernel_tf = np.transpose(kernel, (1, 0))
        if self.use_bias:
            self.set_weights([kernel_tf, bias])
        else:
            self.set_weights([kernel_tf])


def manual_bilinear_grid_sample_nhwc(inputs, grid):
    """Pure TensorFlow bilinear grid sampler matching PyTorch align_corners=True/zeros.
    Optimized for TFLite compatibility and index safety.
    """
    inputs = tf.convert_to_tensor(inputs)
    grid = tf.convert_to_tensor(grid, dtype=inputs.dtype)

    input_shape = tf.shape(inputs)
    n = input_shape[0]
    h_in = input_shape[1]
    w_in = input_shape[2]
    
    grid_shape = tf.shape(grid)
    h_out = grid_shape[1]
    w_out = grid_shape[2]

    # Split grid into x, y normalized coordinates [-1, 1]
    grid_split = tf.split(grid, 2, axis=-1)
    x_norm = tf.squeeze(grid_split[0], axis=-1)
    y_norm = tf.squeeze(grid_split[1], axis=-1)

    # Scale to [0, h_in-1] and [0, w_in-1]
    h_in_f = tf.cast(h_in, inputs.dtype)
    w_in_f = tf.cast(w_in, inputs.dtype)
    x = (x_norm + 1.0) * 0.5 * (w_in_f - 1.0)
    y = (y_norm + 1.0) * 0.5 * (h_in_f - 1.0)

    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    # Bilinear weights
    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    # padding_mode='zeros': invalid corner samples contribute zero weight.
    max_x = w_in_f - 1.0
    max_y = h_in_f - 1.0
    
    valid00 = (x0 >= 0.0) & (x0 <= max_x) & (y0 >= 0.0) & (y0 <= max_y)
    valid01 = (x1 >= 0.0) & (x1 <= max_x) & (y0 >= 0.0) & (y0 <= max_y)
    valid10 = (x0 >= 0.0) & (x0 <= max_x) & (y1 >= 0.0) & (y1 <= max_y)
    valid11 = (x1 >= 0.0) & (x1 <= max_x) & (y1 >= 0.0) & (y1 <= max_y)

    w00 = tf.where(valid00, wy0 * wx0, 0.0)
    w01 = tf.where(valid01, wy0 * wx1, 0.0)
    w10 = tf.where(valid10, wy1 * wx0, 0.0)
    w11 = tf.where(valid11, wy1 * wx1, 0.0)
    
    # Clamp indices to [0, size-1] for safe gather
    x0_idx = tf.cast(tf.clip_by_value(x0, 0.0, max_x), tf.int32)
    x1_idx = tf.cast(tf.clip_by_value(x1, 0.0, max_x), tf.int32)
    y0_idx = tf.cast(tf.clip_by_value(y0, 0.0, max_y), tf.int32)
    y1_idx = tf.cast(tf.clip_by_value(y1, 0.0, max_y), tf.int32)

    # Batch indices
    batch_idx = tf.range(n, dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, [n, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, h_out, w_out])
    
    def get_val(yi, xi):
        idx = tf.stack([batch_idx, yi, xi], axis=-1)
        return tf.gather_nd(inputs, idx)

    v00 = get_val(y0_idx, x0_idx)
    v01 = get_val(y0_idx, x1_idx)
    v10 = get_val(y1_idx, x0_idx)
    v11 = get_val(y1_idx, x1_idx)
    
    w00 = tf.expand_dims(w00, -1)
    w01 = tf.expand_dims(w01, -1)
    w10 = tf.expand_dims(w10, -1)
    w11 = tf.expand_dims(w11, -1)
    
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11


class TFDAttentionBaseline(tf.keras.layers.Layer):
    """NHWC equivalent of export-safe PyTorch `DAttentionBaseline`."""

    def __init__(
        self,
        q_size=(14, 14),
        kv_size=(14, 14),
        n_heads: int = 16,
        n_head_channels: int = 32,
        n_groups: int = 4,
        stride: int = 1,
        offset_range_factor: int = 2,
        use_pe: bool = True,
        dwc_pe: bool = False,
        no_off: bool = False,
        fixed_pe: bool = False,
        stage_idx: int = 2,
        epsilon: float = 1e-5,
        name: str = "tf_dattention_baseline",
    ) -> None:
        super().__init__(name=name)
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        self.n_groups = n_groups
        self.n_group_heads = n_heads // n_groups
        self.n_group_channels = n_head_channels * self.n_group_heads
        self.stride = stride
        self.offset_range_factor = offset_range_factor
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe
        self.stage_idx = stage_idx
        self.epsilon = epsilon

        self.nc = n_heads * n_head_channels
        self.scale = n_head_channels**-0.5

        # Layers
        self.proj_q = PytorchLikeConv1x1(self.nc, name="proj_q")
        self.proj_k = PytorchLikeConv1x1(self.nc, name="proj_k")
        self.proj_v = PytorchLikeConv1x1(self.nc, name="proj_v")
        self.proj_out = PytorchLikeConv1x1(self.nc, name="proj_out")

        # Offset branch
        off_kernel_size = 5 if stage_idx <= 2 else 3
        self.offset_depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=off_kernel_size,
            strides=1,
            padding="same",
            use_bias=True,
            name="offset_depthwise",
        )
        self.offset_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name="offset_norm")
        self.offset_pointwise = PytorchLikeConv1x1(2, use_bias=False, name="offset_pointwise")

        # Reference points and gates
        self.ref_point14 = PytorchLikeConv1x1(2, use_bias=True, name="ref_point")
        self.ref_gate = PytorchLikeConv1x1(1, use_bias=True, name="ref_gate")

        # RPE Table
        self.rpe_table = self.add_weight(
            name="rpe_table",
            shape=(n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x, training: bool = False):
        output, _debug = self.call_with_debug(x, training=training)
        return output

    def call_with_debug(self, x, training: bool = False):
        del training
        shape = tf.shape(x)
        batch = shape[0]
        height = shape[1]
        width = shape[2]

        q = self.proj_q(x)
        
        # Offset branch uses q rearranged to groups
        q_off = self._group_channels_to_batch(q)
        offset = self.offset_depthwise(q_off)
        offset = self.offset_norm(offset)
        offset = tf.nn.gelu(offset, approximate=False)
        offset = self.offset_pointwise(offset) # [B*G, H, W, 2]

        # Offset range scaling
        if self.offset_range_factor > 0:
            hk_f = tf.cast(height, x.dtype)
            wk_f = tf.cast(width, x.dtype)
            offset_range = tf.stack([1.0 / hk_f, 1.0 / wk_f]) # [2]
            offset_range = tf.reshape(offset_range, [1, 1, 1, 2])
            offset = tf.tanh(offset) * offset_range * tf.cast(self.offset_range_factor, x.dtype)

        reference = tf.tanh(self.ref_point14(x)) # [B, H, W, 2]
        gate = tf.sigmoid(self.ref_gate(x) * -99999.0) # [B, H, W, 1]
        
        # repeat for groups
        reference_grouped = tf.repeat(reference, repeats=self.n_groups, axis=0)
        gate_grouped = tf.repeat(gate, repeats=self.n_groups, axis=0)

        if self.no_off:
            offset = tf.zeros_like(offset)

        # Deformable grid logic
        offset_y, offset_x = tf.split(offset, 2, axis=-1)
        reference_y, reference_x = tf.split(reference_grouped, 2, axis=-1)
        
        temp_x = tf.where(offset_x * reference_x <= 0.0, tf.zeros_like(offset_x), tf.ones_like(offset_x))
        temp_y = tf.where(offset_y * reference_y <= 0.0, tf.zeros_like(offset_y), tf.ones_like(offset_y))
        offset_temp = temp_x * temp_y
        manu_offset = tf.where(offset_temp <= 0.0, tf.fill(tf.shape(offset_temp), tf.cast(0.25, x.dtype)), tf.ones_like(offset_temp))
        manu_offset = tf.concat([manu_offset, manu_offset], axis=-1)
        offset_final = offset * manu_offset

        if self.offset_range_factor >= 0:
            pos = (offset_final + reference_grouped) * gate_grouped
        else:
            pos = tf.tanh((offset_final + reference_grouped) * gate_grouped)

        pos_y, pos_x = tf.split(pos, 2, axis=-1)
        grid = tf.concat([pos_x, pos_y], axis=-1)
        
        x_grouped = self._group_channels_to_batch(x)
        x_sampled_grouped = manual_bilinear_grid_sample_nhwc(x_grouped, grid)
        x_sampled_for_conv = self._sampled_group_to_conv_input(x_sampled_grouped, batch, height, width)

        q_heads = self._conv_map_to_heads(q, batch, height * width)
        k_heads = self._conv_map_to_heads(self.proj_k(x_sampled_for_conv), batch, height * width)
        v_heads = self._conv_map_to_heads(self.proj_v(x_sampled_for_conv), batch, height * width)

        attn = tf.einsum("bcm,bcn->bmn", q_heads, k_heads) * tf.cast(self.scale, x.dtype)
        
        # RPE Bias
        q_grid = self._get_ref_points(height, width, batch, x.dtype)
        displacement = (
            tf.expand_dims(tf.reshape(q_grid, [batch * self.n_groups, height * width, 2]), axis=2)
            - tf.expand_dims(tf.reshape(pos, [batch * self.n_groups, height * width, 2]), axis=1)
        ) * 0.5
        disp_y, disp_x = tf.split(displacement, 2, axis=-1)
        disp_grid = tf.concat([disp_x, disp_y], axis=-1)
        
        rpe_bias = self._rpe_bias_for_sampler(batch, height, width)
        attn_bias_grouped = manual_bilinear_grid_sample_nhwc(rpe_bias, disp_grid)
        attn_bias = self._attn_bias_group_to_heads(attn_bias_grouped, batch, height * width, height * width)
        
        attn = attn + attn_bias
        attn_softmax = tf.nn.softmax(attn, axis=2)

        out_heads = tf.einsum("bmn,bcn->bcm", attn_softmax, v_heads)
        out_map = self._heads_to_map(out_heads, batch, height, width)
        output = self.proj_out(out_map)

        debug = {
            "pos": pos,
            "reference": reference_grouped,
            "gate": gate_grouped,
            "x_sampled_grouped": x_sampled_grouped,
            "attn_bias": attn_bias,
            "attn_softmax": attn_softmax,
            "output": output,
        }
        return output, debug

    def _group_channels_to_batch(self, x):
        shape = tf.shape(x)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        x = tf.reshape(x, [batch, height, width, self.n_groups, self.n_group_channels])
        x = tf.transpose(x, [0, 3, 1, 2, 4])
        return tf.reshape(x, [batch * self.n_groups, height, width, self.n_group_channels])

    def _sampled_group_to_conv_input(self, sampled_grouped, batch, h_key, w_key):
        grouped = tf.reshape(sampled_grouped, [batch, self.n_groups, h_key, w_key, self.n_group_channels])
        grouped = tf.transpose(grouped, [0, 2, 3, 1, 4])
        sampled = tf.reshape(grouped, [batch, h_key * w_key, self.nc])
        return tf.expand_dims(sampled, axis=1)

    def _conv_map_to_heads(self, x, batch, n_positions):
        x = tf.reshape(x, [batch, -1, n_positions, self.n_heads, self.n_head_channels])
        x = tf.transpose(x, [0, 3, 4, 1, 2])
        return tf.reshape(x, [batch * self.n_heads, self.n_head_channels, n_positions])

    def _heads_to_map(self, x, batch, height, width):
        x = tf.reshape(x, [batch, self.n_heads, self.n_head_channels, height, width])
        x = tf.transpose(x, [0, 3, 4, 1, 2])
        return tf.reshape(x, [batch, height, width, self.nc])

    def _get_ref_points(self, height, width, batch, dtype):
        ref_y = (tf.cast(tf.range(height), dtype) + 0.5) / tf.cast(height, dtype) * 2.0 - 1.0
        ref_x = (tf.cast(tf.range(width), dtype) + 0.5) / tf.cast(width, dtype) * 2.0 - 1.0
        yy, xx = tf.meshgrid(ref_y, ref_x, indexing="ij")
        ref = tf.stack([yy, xx], axis=-1)
        ref = tf.expand_dims(ref, axis=0)
        return tf.tile(ref, [batch * self.n_groups, 1, 1, 1])

    def _rpe_bias_for_sampler(self, batch, height, width):
        rpe = tf.reshape(self.rpe_table, [self.n_groups, self.n_group_heads, 2 * height - 1, 2 * width - 1])
        rpe = tf.expand_dims(rpe, axis=0)
        rpe = tf.tile(rpe, [batch, 1, 1, 1, 1])
        rpe = tf.transpose(rpe, [0, 1, 3, 4, 2])
        return tf.reshape(rpe, [batch * self.n_groups, 2 * height - 1, 2 * width - 1, self.n_group_heads])

    def _attn_bias_group_to_heads(self, attn_bias_grouped, batch, hw, n_sample):
        attn_bias = tf.reshape(attn_bias_grouped, [batch, self.n_groups, hw, n_sample, self.n_group_heads])
        attn_bias = tf.transpose(attn_bias, [0, 1, 4, 2, 3])
        return tf.reshape(attn_bias, [batch * self.n_heads, hw, n_sample])

    def build_for_input_shape(self, input_shape=(1, 14, 14, 512)) -> None:
        self(tf.zeros(input_shape, dtype=tf.float32), training=False)

    def load_from_pytorch_state_dict(self, state_dict, prefix: str = "stages.2.attns.1") -> None:
        depthwise_weight = _tensor_from_state(state_dict, f"{prefix}.conv_offset.0.weight")
        depthwise_kernel = np.transpose(depthwise_weight, (2, 3, 0, 1))
        self.offset_depthwise.set_weights([depthwise_kernel, _tensor_from_state(state_dict, f"{prefix}.conv_offset.0.bias")])
        self.offset_norm.set_weights(
            [
                _tensor_from_state(state_dict, f"{prefix}.conv_offset.1.norm.weight"),
                _tensor_from_state(state_dict, f"{prefix}.conv_offset.1.norm.bias"),
            ]
        )
        self.offset_pointwise.set_from_pytorch(_tensor_from_state(state_dict, f"{prefix}.conv_offset.3.weight"))
        for name in ["proj_q", "proj_k", "proj_v", "proj_out", "ref_point14", "ref_gate"]:
            getattr(self, name).set_from_pytorch(
                _tensor_from_state(state_dict, f"{prefix}.{name}.weight"),
                _tensor_from_state(state_dict, f"{prefix}.{name}.bias"),
            )
        self.rpe_table.assign(_tensor_from_state(state_dict, f"{prefix}.rpe_table"))

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "target_pytorch_module": "stages.2.attns.1",
            "input_layout": "NHWC/channels_last",
            "q_size": [self.q_h, self.q_w],
            "heads": self.n_heads,
            "head_channels": self.n_head_channels,
            "groups": self.n_groups,
            "group_channels": self.n_group_channels,
            "offset_branch": [
                "DepthwiseConv2D kernel 5 stride 1 padding same, mapped from grouped Conv2d",
                "LayerNormalization(axis=-1, epsilon=1e-5)",
                "exact GELU",
                "PytorchLikeConv1x1 to 2 channels without bias",
            ],
            "standard_1x1_projections": [
                "PytorchLikeConv1x1 for proj_q/proj_k/proj_v/proj_out",
            ],
            "sampling": [
                "sigmoid(ref_point14) for reference points",
                "sigmoid(ref_gate) for gate values",
                "offset * manu_offset + reference for grid coordinates",
                "manual_bilinear_grid_sample_nhwc for sampling features and RPE bias",
            ],
            "attention": "scaled dot-product multi-head attention with groups and relative position bias",
            "rpe_table_shape": [self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1],
        }
