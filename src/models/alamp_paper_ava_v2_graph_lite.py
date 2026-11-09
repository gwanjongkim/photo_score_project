"""A-LAMP-paper-AVA-v2_graph_lite approximation model.

This is an object/global attribute graph approximation, not an official A-LAMP reproduction.
"""

from __future__ import annotations
import sys
from typing import Any
import tensorflow as tf

from src.models.alamp_paper_ava import (
    MergePatchBatch,
    RestorePatchBatch,
    Vgg16UnitPreprocess,
    _normalize_backbone_weights,
    _build_global_vgg16_branch,
    _build_fixed_patch_sequence_features,
)

MODEL_VARIANT = "A-LAMP-paper-AVA-v2_graph_lite"
MODEL_DESCRIPTION = "A-LAMP-paper-oriented approximation with graph-lite branch"
STYLE_DESCRIPTION = "A-LAMP-style AVA classification model"
_UNSET = object()

def _model_cfg(config: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not config:
        return default
    model_section = config.get("model", {})
    if isinstance(model_section, dict) and key in model_section:
        return model_section[key]
    return config.get(key, default)

def _bool_cfg(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)

@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class GraphLiteBranch(tf.keras.layers.Layer):
    def __init__(self, max_objects: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_objects = int(max_objects)
        self.dropout_rate = float(dropout)
        self.class_embed = tf.keras.layers.Embedding(input_dim=80, output_dim=16, name="class_embedding")
        self.node_dense1 = tf.keras.layers.Dense(64, activation="relu", name="node_dense1")
        self.node_dropout = tf.keras.layers.Dropout(self.dropout_rate, name="node_dropout")
        self.node_dense2 = tf.keras.layers.Dense(64, activation="relu", name="node_dense2")

    def call(self, inputs: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        object_mask = tf.cast(inputs["object_mask"], tf.float32)
        boxes = tf.cast(inputs["boxes_norm_xyxy"], tf.float32)
        centers = tf.cast(inputs["centers_norm_xy"], tf.float32)
        area = tf.cast(inputs["area_ratio"], tf.float32)
        class_ids = tf.cast(inputs["class_ids"], tf.int32)
        conf = tf.cast(inputs["confidences"], tf.float32)
        local_edges = tf.cast(inputs["local_edges"], tf.float32)
        global_edges = tf.cast(inputs["global_edges"], tf.float32)

        area_exp = tf.expand_dims(area, axis=-1)
        conf_exp = tf.expand_dims(conf, axis=-1)
        node_geom = tf.concat([boxes, centers, area_exp, conf_exp], axis=-1)
        
        class_emb = self.class_embed(class_ids)
        node_feat = tf.concat([node_geom, class_emb], axis=-1)
        
        neighbor_mask = tf.reshape(object_mask, [-1, 1, self.max_objects, 1])
        masked_local_edges = local_edges * neighbor_mask
        local_edge_feat = tf.reduce_sum(masked_local_edges, axis=2)

        combined_node = tf.concat([node_feat, local_edge_feat, global_edges], axis=-1)
        
        x = self.node_dense1(combined_node)
        x = self.node_dropout(x, training=training)
        x = self.node_dense2(x)
        
        node_mask = tf.expand_dims(object_mask, axis=-1)
        masked_x = x * node_mask
        
        sum_nodes = tf.reduce_sum(masked_x, axis=1)
        sum_mask = tf.reduce_sum(node_mask, axis=1)
        
        return sum_nodes / tf.maximum(sum_mask, 1e-6)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "max_objects": self.max_objects, "dropout": self.dropout_rate}

def build_alamp_paper_ava_v2_model(
    config: dict[str, Any] | None = None,
    *,
    patch_count: int | None = None,
    image_size_patch: int | None = None,
    image_size_global: int | None = None,
    max_objects: int | None = None,
    backbone: str | None = None,
    backbone_weights: str | None | object = _UNSET,
    train_backbone: bool | None = None,
    feature_dim: int | None = None,
    dropout: float | None = None,
) -> tf.keras.Model:
    patch_count = int(patch_count or _model_cfg(config, "patch_count", 5))
    image_size_patch = int(image_size_patch or _model_cfg(config, "image_size_patch", _model_cfg(config, "patch_size", 224)))
    image_size_global = int(image_size_global or _model_cfg(config, "image_size_global", _model_cfg(config, "global_size", 384)))
    max_objects = int(max_objects or _model_cfg(config, "max_objects", 4))
    
    backbone = str(backbone or _model_cfg(config, "backbone", "VGG16"))
    backbone_weights_value = _model_cfg(config, "backbone_weights", "imagenet") if backbone_weights is _UNSET else backbone_weights
    backbone_weights = _normalize_backbone_weights(backbone_weights_value if isinstance(backbone_weights_value, str) else None)
    
    train_backbone = _bool_cfg(train_backbone if train_backbone is not None else _model_cfg(config, "train_backbone", False))
    feature_dim = int(feature_dim or _model_cfg(config, "feature_dim", 256))
    dropout = float(dropout if dropout is not None else _model_cfg(config, "dropout", 0.3))

    inputs = {}
    patch_input = tf.keras.Input(shape=(patch_count, image_size_patch, image_size_patch, 3), name="patches")
    global_input = tf.keras.Input(shape=(image_size_global, image_size_global, 3), name="global_view")
    inputs["patches"] = patch_input
    inputs["global_view"] = global_input

    object_mask = tf.keras.Input(shape=(max_objects,), dtype=tf.float32, name="object_mask")
    boxes_norm_xyxy = tf.keras.Input(shape=(max_objects, 4), dtype=tf.float32, name="boxes_norm_xyxy")
    centers_norm_xy = tf.keras.Input(shape=(max_objects, 2), dtype=tf.float32, name="centers_norm_xy")
    area_ratio = tf.keras.Input(shape=(max_objects,), dtype=tf.float32, name="area_ratio")
    class_ids = tf.keras.Input(shape=(max_objects,), dtype=tf.int32, name="class_ids")
    confidences = tf.keras.Input(shape=(max_objects,), dtype=tf.float32, name="confidences")
    local_edges = tf.keras.Input(shape=(max_objects, max_objects, 3), dtype=tf.float32, name="local_edges")
    global_edges = tf.keras.Input(shape=(max_objects, 3), dtype=tf.float32, name="global_edges")
    
    inputs.update({
        "object_mask": object_mask,
        "boxes_norm_xyxy": boxes_norm_xyxy,
        "centers_norm_xy": centers_norm_xy,
        "area_ratio": area_ratio,
        "class_ids": class_ids,
        "confidences": confidences,
        "local_edges": local_edges,
        "global_edges": global_edges,
    })

    patch_features = _build_fixed_patch_sequence_features(
        patch_input=patch_input,
        patch_count=patch_count,
        patch_size=image_size_patch,
        backbone_weights=backbone_weights,
        train_backbone=train_backbone,
        feature_dim=feature_dim,
        dropout=dropout,
    )
    patch_summary = tf.keras.layers.GlobalAveragePooling1D(name="patch_feature_average")(patch_features)
    patch_summary = tf.keras.layers.Dense(feature_dim, activation="relu", name="patch_subnet_dense")(patch_summary)

    global_features = _build_global_vgg16_branch(
        global_input=global_input,
        global_size=image_size_global,
        backbone_weights=backbone_weights,
        train_backbone=train_backbone,
        feature_dim=feature_dim,
        dropout=dropout,
    )

    graph_features = GraphLiteBranch(max_objects=max_objects, dropout=dropout, name="graph_lite_branch")({
        "object_mask": object_mask,
        "boxes_norm_xyxy": boxes_norm_xyxy,
        "centers_norm_xy": centers_norm_xy,
        "area_ratio": area_ratio,
        "class_ids": class_ids,
        "confidences": confidences,
        "local_edges": local_edges,
        "global_edges": global_edges,
    })

    fused = tf.keras.layers.Concatenate(name="alamp_paper_ava_v2_fusion")(
        [patch_summary, global_features, graph_features]
    )
    
    fused = tf.keras.layers.Dropout(dropout, name="classifier_dropout1")(fused)
    fused = tf.keras.layers.Dense(256, activation="relu", name="classifier_dense")(fused)
    fused = tf.keras.layers.Dropout(max(0.1, dropout * 0.5), name="classifier_dropout2")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(fused)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="alamp_paper_ava_v2_graph_lite")
    model._alamp_paper_ava_config = {
        "model_variant": MODEL_VARIANT,
        "description": MODEL_DESCRIPTION,
        "style": STYLE_DESCRIPTION,
        "backbone": backbone,
        "patch_count": patch_count,
        "image_size_patch": image_size_patch,
        "image_size_global": image_size_global,
        "max_objects": max_objects,
    }
    return model
