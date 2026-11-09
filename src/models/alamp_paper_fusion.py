# A-LAMP 논문 지향 MPNet-그래프 융합 실험 모델을 정의한다.
"""Research-only MPNet + layout graph fusion models.

These models are paper-oriented approximations for ablation experiments. They do
not replace the practical app-facing A-LAMP/RGNet models.
"""

from __future__ import annotations

import tensorflow as tf


class GraphConvolutionLayer(tf.keras.layers.Layer):
    """Small masked graph-convolution layer using row-normalized adjacency."""

    def __init__(
        self,
        units: int,
        activation: str | None = "relu",
        dropout_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = tf.keras.activations.get(activation)
        self.activation_name = activation
        self.dropout_rate = float(dropout_rate)
        self.projection = tf.keras.layers.Dense(self.units)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training: bool = False):
        node_features, adjacency, object_mask = inputs
        node_features = tf.cast(node_features, tf.float32)
        adjacency = tf.cast(adjacency, tf.float32)
        object_mask = tf.cast(object_mask, tf.float32)

        adjacency = tf.where(tf.math.is_finite(adjacency), adjacency, 0.0)
        adjacency = tf.maximum(adjacency, 0.0)

        pair_mask = object_mask[:, :, None] * object_mask[:, None, :]
        adjacency = adjacency * pair_mask
        adjacency = adjacency + tf.linalg.diag(object_mask)

        degree = tf.reduce_sum(adjacency, axis=-1, keepdims=True)
        adjacency_norm = adjacency / tf.maximum(degree, 1.0e-6)

        hidden = tf.matmul(adjacency_norm, node_features)
        hidden = self.projection(hidden)
        if self.activation is not None:
            hidden = self.activation(hidden)

        hidden = hidden * object_mask[:, :, None]
        hidden = self.dropout(hidden, training=training)
        return tf.where(tf.math.is_finite(hidden), hidden, 0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation_name,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class MaskedMeanPoolingLayer(tf.keras.layers.Layer):
    """Mask-aware mean pooling over object nodes."""

    def call(self, inputs):
        node_features, object_mask = inputs
        node_features = tf.cast(node_features, tf.float32)
        object_mask = tf.cast(object_mask, tf.float32)

        masked = node_features * object_mask[:, :, None]
        denom = tf.maximum(tf.reduce_sum(object_mask, axis=1, keepdims=True), 1.0)
        pooled = tf.reduce_sum(masked, axis=1) / denom
        return tf.where(tf.math.is_finite(pooled), pooled, 0.0)


def build_mpnet_graphlite_fusion_model(
    mpnet_feature_dim: int,
    graph_vector_dim: int,
    graph_hidden_dim: int = 128,
    fusion_hidden_dim: int = 128,
    dropout_rate: float = 0.25,
) -> tf.keras.Model:
    """Build MPNet-feature + fixed graph-vector MLP fusion."""

    mpnet_input = tf.keras.Input(shape=(mpnet_feature_dim,), name="mpnet_features")
    graph_input = tf.keras.Input(shape=(graph_vector_dim,), name="graph_vector")

    graph_branch = tf.keras.layers.Dense(
        graph_hidden_dim, activation="relu", name="graphlite_dense1"
    )(graph_input)
    graph_branch = tf.keras.layers.Dropout(dropout_rate, name="graphlite_dropout")(
        graph_branch
    )
    graph_branch = tf.keras.layers.Dense(
        graph_hidden_dim // 2, activation="relu", name="graphlite_dense2"
    )(graph_branch)

    mpnet_branch = tf.keras.layers.Dense(
        fusion_hidden_dim, activation="relu", name="mpnet_feature_projection"
    )(mpnet_input)

    fused = tf.keras.layers.Concatenate(name="fusion_concat")(
        [mpnet_branch, graph_branch]
    )
    fused = tf.keras.layers.Dense(
        fusion_hidden_dim, activation="relu", name="fusion_dense"
    )(fused)
    fused = tf.keras.layers.Dropout(dropout_rate, name="fusion_dropout")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(fused)

    return tf.keras.Model(
        inputs={"mpnet_features": mpnet_input, "graph_vector": graph_input},
        outputs=output,
        name="alamp_paper_mpnet_graphlite_fusion",
    )


def build_mpnet_graphgcn_fusion_model(
    mpnet_feature_dim: int,
    node_feature_dim: int,
    max_objects: int = 4,
    gcn_hidden_dim: int = 64,
    fusion_hidden_dim: int = 128,
    dropout_rate: float = 0.25,
    gcn_layers: int = 2,
) -> tf.keras.Model:
    """Build MPNet-feature + small masked GraphGCN fusion."""

    mpnet_input = tf.keras.Input(shape=(mpnet_feature_dim,), name="mpnet_features")
    node_input = tf.keras.Input(
        shape=(max_objects, node_feature_dim), name="node_features"
    )
    adjacency_input = tf.keras.Input(shape=(max_objects, max_objects), name="adjacency")
    mask_input = tf.keras.Input(shape=(max_objects,), name="object_mask")

    graph_hidden = node_input
    for layer_index in range(max(int(gcn_layers), 1)):
        graph_hidden = GraphConvolutionLayer(
            gcn_hidden_dim,
            activation="relu",
            dropout_rate=dropout_rate if layer_index > 0 else 0.0,
            name=f"graph_conv{layer_index + 1}",
        )([graph_hidden, adjacency_input, mask_input])

    graph_branch = MaskedMeanPoolingLayer(name="graph_masked_mean")(
        [graph_hidden, mask_input]
    )
    graph_branch = tf.keras.layers.Dense(
        fusion_hidden_dim // 2, activation="relu", name="graphgcn_projection"
    )(graph_branch)

    mpnet_branch = tf.keras.layers.Dense(
        fusion_hidden_dim, activation="relu", name="mpnet_feature_projection"
    )(mpnet_input)

    fused = tf.keras.layers.Concatenate(name="fusion_concat")(
        [mpnet_branch, graph_branch]
    )
    fused = tf.keras.layers.Dense(
        fusion_hidden_dim, activation="relu", name="fusion_dense"
    )(fused)
    fused = tf.keras.layers.Dropout(dropout_rate, name="fusion_dropout")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(fused)

    return tf.keras.Model(
        inputs={
            "mpnet_features": mpnet_input,
            "node_features": node_input,
            "adjacency": adjacency_input,
            "object_mask": mask_input,
        },
        outputs=output,
        name="alamp_paper_mpnet_graphgcn_fusion",
    )


def get_alamp_paper_fusion_custom_objects() -> dict[str, object]:
    return {
        "GraphConvolutionLayer": GraphConvolutionLayer,
        "MaskedMeanPoolingLayer": MaskedMeanPoolingLayer,
    }
