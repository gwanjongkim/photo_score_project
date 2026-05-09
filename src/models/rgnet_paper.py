# RGNet 논문 지향 AADB 회귀 모델을 정의한다.
"""Paper-oriented RGNet AADB regression model.

This module is an isolated paper-comparison track. It does not replace the
practical RGNet implementation in ``src/models/rgnet.py``.

Variant:
    RGNet-paper-v0 approximation.

Implemented paper-oriented pieces:
    - DenseNet121 fully convolutional image feature backbone.
    - Spatial feature-map positions are treated as region nodes.
    - Dense cosine-similarity region adjacency.
    - Softmax-normalized adjacency.
    - Residual graph convolution blocks over region nodes.
    - Scalar sigmoid regression head for AADB scores in [0, 1].

Not an official reproduction:
    The exact RegionGraph construction details from the paper are ambiguous
    from local evidence, so this is a documented v0 approximation rather than
    the authors' official model or weights.
"""

from __future__ import annotations

import sys

import tensorflow as tf

MODEL_VARIANT = "RGNet-paper-v0 approximation"


def _normalize_backbone_weights(weights: str | None) -> str | None:
    if weights is None:
        return None
    if str(weights).lower() in {"none", "null", "false", "random"}:
        return None
    return str(weights)


def _build_densenet121_backbone(
    input_shape: tuple[int, int, int],
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    weights = _normalize_backbone_weights(weights)
    try:
        return tf.keras.applications.DenseNet121(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    except Exception as exc:
        if weights is not None:
            print(
                f"DenseNet121 {weights} weights unavailable, falling back to random init: {exc}",
                file=sys.stderr,
            )
            return tf.keras.applications.DenseNet121(
                include_top=False,
                weights=None,
                input_shape=input_shape,
            )
        raise


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class DenseNetUnitPreprocess(tf.keras.layers.Layer):
    """Converts [0, 1] RGB tensors to DenseNet121 ImageNet preprocessing."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        return tf.keras.applications.densenet.preprocess_input(inputs * 255.0)

    def get_config(self) -> dict[str, object]:
        return dict(super().get_config())


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class RegionSimilarityAdjacency(tf.keras.layers.Layer):
    """Builds softmax-normalized dense adjacency from cosine region similarity."""

    def __init__(self, temperature: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def call(self, node_features: tf.Tensor) -> tf.Tensor:
        node_features = tf.cast(node_features, tf.float32)
        normalized = tf.math.l2_normalize(node_features, axis=-1)
        similarity = tf.matmul(normalized, normalized, transpose_b=True)
        return tf.nn.softmax(similarity / self.temperature, axis=-1)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "temperature": self.temperature}


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class ResidualGraphConvolution(tf.keras.layers.Layer):
    """Residual graph convolution over dense region-node adjacency."""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation_name = activation
        self.dropout_rate = float(dropout)
        self.proj = tf.keras.layers.Dense(self.units, use_bias=False, name="node_projection")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.activation = tf.keras.activations.get(activation)
        self.residual_proj: tf.keras.layers.Layer | None = None

    def build(self, input_shape):
        node_shape = input_shape[0]
        if int(node_shape[-1]) != self.units:
            self.residual_proj = tf.keras.layers.Dense(
                self.units,
                use_bias=False,
                name="residual_projection",
            )
        super().build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        node_features, adjacency = inputs
        node_features = tf.cast(node_features, tf.float32)
        adjacency = tf.cast(adjacency, tf.float32)
        residual = node_features if self.residual_proj is None else self.residual_proj(node_features)
        x = tf.matmul(adjacency, node_features)
        x = self.proj(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return self.norm(x + residual)

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "units": self.units,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
        }


def build_rgnet_paper_model(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    backbone_weights: str | None = "imagenet",
    region_dim: int = 256,
    graph_units: int = 256,
    graph_blocks: int = 2,
    graph_temperature: float = 0.25,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Builds the isolated RGNet-paper-v0 approximation for AADB regression."""

    backbone = _build_densenet121_backbone(input_shape=input_shape, weights=backbone_weights)
    backbone.trainable = True

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = DenseNetUnitPreprocess(name="densenet_preprocess")(inputs)
    feature_map = backbone(x, training=False)
    feature_map = tf.keras.layers.Conv2D(
        int(region_dim),
        1,
        padding="same",
        activation="relu",
        name="region_feature_projection",
    )(feature_map)

    node_features = tf.keras.layers.Reshape((-1, int(region_dim)), name="region_nodes")(feature_map)
    adjacency = RegionSimilarityAdjacency(
        temperature=graph_temperature,
        name="region_similarity_adjacency",
    )(node_features)

    graph_context = node_features
    for block_index in range(int(graph_blocks)):
        graph_context = ResidualGraphConvolution(
            units=int(graph_units),
            dropout=0.1,
            name=f"residual_graph_convolution_{block_index + 1}",
        )((graph_context, adjacency))

    graph_pool = tf.keras.layers.GlobalAveragePooling1D(name="region_graph_pool")(graph_context)
    spatial_pool = tf.keras.layers.GlobalAveragePooling2D(name="spatial_feature_pool")(feature_map)
    fused = tf.keras.layers.Concatenate(name="graph_spatial_fusion")([graph_pool, spatial_pool])
    fused = tf.keras.layers.Dropout(float(dropout), name="head_dropout")(fused)
    fused = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(fused)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="score")(fused)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="rgnet_paper_v0_aadb_regression")


def get_rgnet_paper_custom_objects() -> dict[str, object]:
    return {
        "DenseNetUnitPreprocess": DenseNetUnitPreprocess,
        "RegionSimilarityAdjacency": RegionSimilarityAdjacency,
        "ResidualGraphConvolution": ResidualGraphConvolution,
    }
