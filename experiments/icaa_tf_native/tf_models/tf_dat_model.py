# ICAA17K DAT 전체 추론 경로를 TensorFlow/Keras로 조립한 모듈
from __future__ import annotations

import numpy as np
import tensorflow as tf

from tf_models.patch_projection import TFPatchProjection
from tf_models.soft_histogram import TFSoftHistogram
from tf_models.transformer_stage import (
    TFTransformerStage0,
    TFTransformerStage1,
    TFTransformerStage2,
    TFTransformerStage3,
)


def _tensor_from_state(state_dict, name: str):
    return state_dict[name].detach().cpu().numpy()


def _set_dense_from_linear(layer: tf.keras.layers.Dense, weight, bias) -> None:
    layer.set_weights([np.transpose(weight, (1, 0)), bias])


class TFDownProjection(tf.keras.layers.Layer):
    """NHWC equivalent of PyTorch DAT down_projs[i] for use_conv_patches=False."""

    def __init__(
        self,
        out_channels: int,
        epsilon: float = 1e-5,
        name: str = "tf_down_projection",
    ) -> None:
        super().__init__(name=name)
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="valid",
            use_bias=False,
            name="conv",
        )
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name="norm")

    def call(self, inputs, training: bool = False):
        del training
        return self.norm(self.conv(inputs))

    def load_from_pytorch_state_dict(self, state_dict, prefix: str) -> None:
        conv_weight = _tensor_from_state(state_dict, f"{prefix}.0.weight")
        conv_kernel = np.transpose(conv_weight, (2, 3, 1, 0))
        self.conv.set_weights([conv_kernel])
        self.norm.set_weights(
            [
                _tensor_from_state(state_dict, f"{prefix}.1.norm.weight"),
                _tensor_from_state(state_dict, f"{prefix}.1.norm.bias"),
            ]
        )

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "conv": {
                "type": "tf.keras.layers.Conv2D",
                "filters": self.out_channels,
                "kernel_size": [2, 2],
                "strides": [2, 2],
                "padding": "valid",
                "use_bias": False,
            },
            "norm": {
                "type": "tf.keras.layers.LayerNormalization",
                "axis": -1,
                "epsilon": self.epsilon,
            },
        }


class TFICAA17KDAT(tf.keras.Model):
    """Inference-only NHWC TensorFlow equivalent of the ICAA17K DAT forward path."""

    def __init__(
        self,
        epsilon: float = 1e-5,
        name: str = "tf_icaa17k_dat",
    ) -> None:
        super().__init__(name=name)
        self.epsilon = epsilon
        self.patch_proj = TFPatchProjection(dim_stem=128, patch_size=4, epsilon=epsilon, name="patch_proj")
        self.stage0 = TFTransformerStage0(dim=128, heads=4, window_size=7, expansion=4, epsilon=epsilon, name="stage0")
        self.down_proj0 = TFDownProjection(out_channels=256, epsilon=epsilon, name="down_proj0")
        self.stage1 = TFTransformerStage1(dim=256, heads=8, window_size=7, expansion=4, epsilon=epsilon, name="stage1")
        self.down_proj1 = TFDownProjection(out_channels=512, epsilon=epsilon, name="down_proj1")
        self.stage2 = TFTransformerStage2(dim=512, heads=16, window_size=7, expansion=4, groups=4, epsilon=epsilon, name="stage2")
        self.down_proj2 = TFDownProjection(out_channels=1024, epsilon=epsilon, name="down_proj2")
        self.stage3 = TFTransformerStage3(dim=1024, heads=32, window_size=7, expansion=4, groups=8, epsilon=epsilon, name="stage3")
        self.cls_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, name="cls_norm")
        self.hst_head = tf.keras.layers.Dense(36, use_bias=True, name="hst_head")
        self.hist_feature = TFSoftHistogram(
            n_features=36,
            n_examples=6,
            num_bins=6,
            quantiles=False,
            name="hist_feature",
        )
        self.class_head = tf.keras.layers.Dense(1, use_bias=True, name="class_head")
        self.class_head2 = tf.keras.layers.Dense(1, use_bias=True, name="class_head2")

    def call(self, inputs, training: bool = False):
        output, _debug = self.call_with_debug(inputs, training=training)
        return output

    def call_with_debug(self, inputs, training: bool = False):
        del training
        x = tf.convert_to_tensor(inputs, dtype=self.compute_dtype)
        debug = {"input": x}

        x = self.patch_proj(x, training=False)
        debug["patch_proj"] = x
        x = self.stage0(x, training=False)
        debug["stage0"] = x
        x = self.down_proj0(x, training=False)
        debug["down_proj0"] = x
        x = self.stage1(x, training=False)
        debug["stage1"] = x
        x = self.down_proj1(x, training=False)
        debug["down_proj1"] = x
        x = self.stage2(x, training=False)
        debug["stage2"] = x
        x = self.down_proj2(x, training=False)
        debug["down_proj2"] = x
        x = self.stage3(x, training=False)
        debug["stage3"] = x

        x = self.cls_norm(x)
        debug["cls_norm"] = x
        pooled = tf.reduce_mean(x, axis=[1, 2])
        debug["pooled_feature"] = pooled

        hist_logits = self.hst_head(pooled)
        debug["hst_head"] = hist_logits
        hist = self.hist_feature(hist_logits, training=False)
        debug["soft_histogram_raw"] = hist
        hist_vector = tf.squeeze(tf.transpose(hist, perm=[2, 1, 0]), axis=2)
        debug["soft_histogram"] = hist_vector

        mos = tf.sigmoid(self.class_head(hist_vector))
        color = tf.sigmoid(self.class_head2(hist_vector))
        debug["mos"] = mos
        debug["color"] = color
        output = tf.concat([mos, color], axis=1)
        debug["final_output"] = output
        return output, debug

    def build_for_input_shape(self, input_shape=(1, 224, 224, 3)) -> None:
        self.patch_proj.build_for_input_shape(input_shape)
        self.stage0.build_for_input_shape((1, 56, 56, 128))
        self.down_proj0(tf.zeros((1, 56, 56, 128), dtype=tf.float32), training=False)
        self.stage1.build_for_input_shape((1, 28, 28, 256))
        self.down_proj1(tf.zeros((1, 28, 28, 256), dtype=tf.float32), training=False)
        self.stage2.build_for_input_shape((1, 14, 14, 512))
        self.down_proj2(tf.zeros((1, 14, 14, 512), dtype=tf.float32), training=False)
        self.stage3.build_for_input_shape((1, 7, 7, 1024))
        self.cls_norm(tf.zeros((1, 7, 7, 1024), dtype=tf.float32))
        self.hst_head(tf.zeros((1, 1024), dtype=tf.float32))
        self.hist_feature(tf.zeros((1, 36), dtype=tf.float32), training=False)
        self.class_head(tf.zeros((1, 216), dtype=tf.float32))
        self.class_head2(tf.zeros((1, 216), dtype=tf.float32))

    def load_from_pytorch_state_dict(self, state_dict) -> None:
        self.patch_proj.load_from_pytorch_state_dict(state_dict)
        self.stage0.load_from_pytorch_state_dict(state_dict, prefix="stages.0")
        self.down_proj0.load_from_pytorch_state_dict(state_dict, prefix="down_projs.0")
        self.stage1.load_from_pytorch_state_dict(state_dict, prefix="stages.1")
        self.down_proj1.load_from_pytorch_state_dict(state_dict, prefix="down_projs.1")
        self.stage2.load_from_pytorch_state_dict(state_dict, prefix="stages.2")
        self.down_proj2.load_from_pytorch_state_dict(state_dict, prefix="down_projs.2")
        self.stage3.load_from_pytorch_state_dict(state_dict, prefix="stages.3")
        self.cls_norm.set_weights(
            [
                _tensor_from_state(state_dict, "cls_norm.norm.weight"),
                _tensor_from_state(state_dict, "cls_norm.norm.bias"),
            ]
        )
        _set_dense_from_linear(
            self.hst_head,
            _tensor_from_state(state_dict, "hst_head.weight"),
            _tensor_from_state(state_dict, "hst_head.bias"),
        )
        self.hist_feature.load_from_pytorch_state_dict(state_dict, prefix="hist_feature.")
        _set_dense_from_linear(
            self.class_head,
            _tensor_from_state(state_dict, "class_head.weight"),
            _tensor_from_state(state_dict, "class_head.bias"),
        )
        _set_dense_from_linear(
            self.class_head2,
            _tensor_from_state(state_dict, "class_head2.weight"),
            _tensor_from_state(state_dict, "class_head2.bias"),
        )

    def structure_summary(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "input_layout": "NHWC/channels_last [B, 224, 224, 3]",
            "inference_only": True,
            "forward_order": [
                "patch_proj",
                "stage0",
                "down_proj0",
                "stage1",
                "down_proj1",
                "stage2",
                "down_proj2",
                "stage3",
                "cls_norm",
                "spatial mean over NHWC H/W",
                "hst_head",
                "TFSoftHistogram",
                "transpose/squeeze histogram to [B, 216]",
                "sigmoid(class_head)",
                "sigmoid(class_head2)",
                "concat [mos, color] -> [B, 2]",
            ],
            "components": {
                "patch_proj": self.patch_proj.structure_summary(),
                "stage0": self.stage0.structure_summary(),
                "down_proj0": self.down_proj0.structure_summary(),
                "stage1": self.stage1.structure_summary(),
                "down_proj1": self.down_proj1.structure_summary(),
                "stage2": self.stage2.structure_summary(),
                "down_proj2": self.down_proj2.structure_summary(),
                "stage3": self.stage3.structure_summary(),
                "soft_histogram": self.hist_feature.structure_summary(),
            },
            "heads": {
                "cls_norm": "LayerNormalization(axis=-1, epsilon=1e-5)",
                "hst_head": "Dense(36), mapped from PyTorch hst_head",
                "class_head": "Dense(1) + sigmoid, mapped from PyTorch class_head",
                "class_head2": "Dense(1) + sigmoid, mapped from PyTorch class_head2",
                "cls_head": "Present in PyTorch state_dict but unused by DAT.forward and intentionally not used in TF forward.",
            },
        }
