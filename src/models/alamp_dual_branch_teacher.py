# A-LAMP 듀얼 브랜치 GCN teacher 프로토타입 모델을 정의한다.
from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.datasets.alamp_dual_branch_dataset import NODE_FEATURE_DIM
from src.models.alamp_multipatch_teacher import (
    MergePatchBatch,
    OrderlessMeanMaxAggregation,
    RestorePatchBatch,
    normalize_backbone_weights,
)
from src.models.alamp_paper_fusion import GraphConvolutionLayer, MaskedMeanPoolingLayer


MODEL_VARIANT = "A-LAMP Dual-Branch GCN teacher prototype"
MODEL_DESCRIPTION = "VGG16 Multi-Patch branch fused with a small YOLO-layout GCN branch"
REPRODUCTION_CLAIM = "not full A-LAMP reproduction"


def build_alamp_dual_branch_teacher_model(
    *,
    patch_count: int = 5,
    patch_size: int = 224,
    max_objects: int = 4,
    node_feature_dim: int = NODE_FEATURE_DIM,
    backbone_weights: str | None = "imagenet",
    backbone_trainable: bool = False,
    patch_feature_units: int = 256,
    gcn_units: int = 64,
    fusion_units: int = 256,
    dropout_rate: float = 0.4,
    gcn_layers: int = 2,
) -> tf.keras.Model:
    backbone_weights = normalize_backbone_weights(backbone_weights)

    patches_input = tf.keras.Input(
        shape=(patch_count, patch_size, patch_size, 3),
        name="patches",
    )
    node_input = tf.keras.Input(
        shape=(max_objects, node_feature_dim),
        name="node_features",
    )
    adjacency_input = tf.keras.Input(
        shape=(max_objects, max_objects),
        name="adjacency",
    )
    mask_input = tf.keras.Input(shape=(max_objects,), name="object_mask")

    patch_branch = MergePatchBatch(
        patch_count=patch_count,
        patch_size=patch_size,
        name="merge_patch_batch",
    )(patches_input)
    backbone = tf.keras.applications.VGG16(
        include_top=False,
        weights=backbone_weights,
        input_shape=(patch_size, patch_size, 3),
        name="vgg16_backbone",
    )
    backbone.trainable = bool(backbone_trainable)
    for layer in backbone.layers:
        layer.trainable = bool(backbone_trainable)

    patch_branch = backbone(patch_branch)
    patch_branch = tf.keras.layers.GlobalAveragePooling2D(name="patch_gap")(patch_branch)
    patch_branch = RestorePatchBatch(
        patch_count=patch_count,
        name="restore_patch_batch",
    )(patch_branch)
    patch_branch = OrderlessMeanMaxAggregation(name="orderless_mean_max_aggregation")(patch_branch)
    patch_branch = tf.keras.layers.Dense(
        patch_feature_units,
        activation="relu",
        name="multipatch_projection",
    )(patch_branch)
    patch_branch = tf.keras.layers.Dropout(dropout_rate, name="multipatch_dropout")(patch_branch)

    graph_branch = node_input
    for index in range(max(int(gcn_layers), 1)):
        graph_branch = GraphConvolutionLayer(
            gcn_units,
            activation="relu",
            dropout_rate=dropout_rate if index > 0 else 0.0,
            name=f"layout_gcn_{index + 1}",
        )([graph_branch, adjacency_input, mask_input])
    graph_branch = MaskedMeanPoolingLayer(name="layout_masked_mean")([graph_branch, mask_input])
    graph_branch = tf.keras.layers.Dense(
        fusion_units // 2,
        activation="relu",
        name="layout_projection",
    )(graph_branch)

    fused = tf.keras.layers.Concatenate(name="dual_branch_concat")([patch_branch, graph_branch])
    fused = tf.keras.layers.Dense(fusion_units, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(dropout_rate, name="fusion_dropout")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(fused)

    model = tf.keras.Model(
        inputs={
            "patches": patches_input,
            "node_features": node_input,
            "adjacency": adjacency_input,
            "object_mask": mask_input,
        },
        outputs=output,
        name="alamp_dual_branch_teacher",
    )
    model._alamp_dual_branch_teacher_config = {  # type: ignore[attr-defined]
        "model_variant": MODEL_VARIANT,
        "description": MODEL_DESCRIPTION,
        "reproduction_claim": REPRODUCTION_CLAIM,
        "patch_count": int(patch_count),
        "patch_size": int(patch_size),
        "backbone": "VGG16",
        "backbone_weights": backbone_weights,
        "backbone_trainable": bool(backbone_trainable),
        "patch_aggregation": "orderless mean features concatenated with orderless max features",
        "max_objects": int(max_objects),
        "node_feature_dim": int(node_feature_dim),
        "gcn_layers": int(gcn_layers),
        "gcn_units": int(gcn_units),
        "fusion_units": int(fusion_units),
        "dropout_rate": float(dropout_rate),
    }
    return model


def summarize_vgg16_trainability(model: tf.keras.Model) -> dict[str, Any]:
    backbone = model.get_layer("vgg16_backbone")
    trainable_layers = [layer.name for layer in backbone.layers if layer.trainable]
    frozen_layers = [layer.name for layer in backbone.layers if not layer.trainable]
    return {
        "backbone_name": backbone.name,
        "trainable": bool(backbone.trainable),
        "trainable_layer_count": len(trainable_layers),
        "frozen_layer_count": len(frozen_layers),
        "trainable_layers": trainable_layers,
        "frozen_layers": frozen_layers,
        "model_trainable_weight_count": len(model.trainable_weights),
        "model_non_trainable_weight_count": len(model.non_trainable_weights),
    }


def get_alamp_dual_branch_teacher_custom_objects() -> dict[str, object]:
    return {
        "MergePatchBatch": MergePatchBatch,
        "RestorePatchBatch": RestorePatchBatch,
        "OrderlessMeanMaxAggregation": OrderlessMeanMaxAggregation,
        "GraphConvolutionLayer": GraphConvolutionLayer,
        "MaskedMeanPoolingLayer": MaskedMeanPoolingLayer,
    }
