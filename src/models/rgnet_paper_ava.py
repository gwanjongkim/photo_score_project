# RGNet 논문 지향 AVA 이진 분류 모델을 정의한다.
"""RGNet-paper-AVA-classification approximation.

This module is isolated from the practical RGNet implementation and from the
AADB paper-oriented regression models. It is not an official paper
reproduction.
"""

from __future__ import annotations

import sys

import tensorflow as tf

MODEL_VARIANT = "RGNet-paper-AVA-classification approximation"

RAW_THEN_SIGMOID = "raw_before_aggregation_then_sigmoid"
SIGMOID_BEFORE_AGGREGATION = "sigmoid_before_aggregation"
SUPPORTED_SCORE_ACTIVATION_MODES = {RAW_THEN_SIGMOID, SIGMOID_BEFORE_AGGREGATION}


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
class DenseNetAvaUnitPreprocess(tf.keras.layers.Layer):
    """Converts [0, 1] RGB tensors to DenseNet121 ImageNet preprocessing."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        return tf.keras.applications.densenet.preprocess_input(inputs * 255.0)

    def get_config(self) -> dict[str, object]:
        return dict(super().get_config())


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class AvaASPPContextModule(tf.keras.layers.Layer):
    """ASPP approximation with parallel dilated convolutions."""

    def __init__(
        self,
        filters: int = 256,
        dilation_rates: tuple[int, ...] = (1, 3, 6, 12, 18),
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.dilation_rates = tuple(int(rate) for rate in dilation_rates)
        self.activation_name = activation
        self.branches: list[tf.keras.Sequential] = []
        for rate in self.dilation_rates:
            self.branches.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(
                            self.filters,
                            3,
                            padding="same",
                            dilation_rate=rate,
                            use_bias=False,
                            name=f"conv_d{rate}",
                        ),
                        tf.keras.layers.BatchNormalization(name=f"bn_d{rate}"),
                        tf.keras.layers.Activation(activation, name=f"act_d{rate}"),
                    ],
                    name=f"aspp_branch_d{rate}",
                )
            )
        self.project = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(self.filters, 1, padding="same", use_bias=False, name="project_conv"),
                tf.keras.layers.BatchNormalization(name="project_bn"),
                tf.keras.layers.Activation(activation, name="project_act"),
            ],
            name="aspp_projection",
        )

    def build(self, input_shape):
        for branch in self.branches:
            branch.build(input_shape)
        concat_shape = list(input_shape)
        concat_shape[-1] = self.filters * len(self.branches)
        self.project.build(tuple(concat_shape))
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        branches = [branch(inputs, training=training) for branch in self.branches]
        x = tf.concat(branches, axis=-1)
        return self.project(x, training=training)

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "filters": self.filters,
            "dilation_rates": self.dilation_rates,
            "activation": self.activation_name,
        }


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class AvaRegionSimilarityAdjacency(tf.keras.layers.Layer):
    """Builds row-wise softmax adjacency from cosine region similarity."""

    def __init__(self, temperature: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def call(self, node_features: tf.Tensor) -> tf.Tensor:
        node_features = tf.cast(node_features, tf.float32)
        normalized = tf.math.l2_normalize(node_features, axis=-1)
        similarity = tf.matmul(normalized, normalized, transpose_b=True)
        temperature = tf.maximum(tf.cast(self.temperature, similarity.dtype), tf.constant(1e-6, similarity.dtype))
        return tf.nn.softmax(similarity / temperature, axis=-1)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "temperature": self.temperature}


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class AvaResidualGraphConvolution(tf.keras.layers.Layer):
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


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class RegionLogitAggregation(tf.keras.layers.Layer):
    """Aggregates per-region logits or probabilities into one image probability."""

    def __init__(
        self,
        aggregation: str = "mean",
        lse_r: float = 4.0,
        score_activation_mode: str = RAW_THEN_SIGMOID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        aggregation = str(aggregation).lower()
        if aggregation not in {"mean", "max", "lse"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        if score_activation_mode not in SUPPORTED_SCORE_ACTIVATION_MODES:
            raise ValueError(f"Unsupported score_activation_mode: {score_activation_mode}")
        self.aggregation = aggregation
        self.lse_r = float(lse_r)
        self.score_activation_mode = score_activation_mode

    def call(self, region_logits: tf.Tensor) -> tf.Tensor:
        region_logits = tf.cast(region_logits, tf.float32)
        values = region_logits
        if self.score_activation_mode == SIGMOID_BEFORE_AGGREGATION:
            values = tf.sigmoid(region_logits)

        if self.aggregation == "mean":
            aggregated = tf.reduce_mean(values, axis=1)
        elif self.aggregation == "max":
            aggregated = tf.reduce_max(values, axis=1)
        else:
            r = tf.maximum(tf.cast(self.lse_r, values.dtype), tf.constant(1e-6, values.dtype))
            scaled = r * values
            max_scaled = tf.reduce_max(scaled, axis=1, keepdims=True)
            stable_mean = tf.reduce_mean(tf.exp(scaled - max_scaled), axis=1, keepdims=True)
            aggregated = tf.squeeze((tf.math.log(tf.maximum(stable_mean, 1e-12)) + max_scaled) / r, axis=1)

        if self.score_activation_mode == RAW_THEN_SIGMOID:
            return tf.sigmoid(aggregated)
        return aggregated

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "aggregation": self.aggregation,
            "lse_r": self.lse_r,
            "score_activation_mode": self.score_activation_mode,
        }


def build_rgnet_paper_ava_model(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    backbone: str = "DenseNet121",
    backbone_weights: str | None = "imagenet",
    region_dim: int = 256,
    graph_units: int = 256,
    graph_blocks: int = 3,
    graph_temperature: float = 0.25,
    graph_dropout: float = 0.1,
    head_dropout: float = 0.3,
    dilation_rates: tuple[int, ...] = (1, 3, 6, 12, 18),
    aggregation: str = "mean",
    lse_r: float = 4.0,
    score_activation_mode: str = RAW_THEN_SIGMOID,
) -> tf.keras.Model:
    """Builds the RGNet-paper-AVA-classification approximation."""

    if str(backbone).lower() != "densenet121":
        raise ValueError(f"Unsupported backbone for this isolated experiment: {backbone}")

    base = _build_densenet121_backbone(input_shape=input_shape, weights=backbone_weights)
    base.trainable = True

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = DenseNetAvaUnitPreprocess(name="densenet_preprocess")(inputs)
    feature_map = base(x, training=False)
    context_map = AvaASPPContextModule(
        filters=int(region_dim),
        dilation_rates=dilation_rates,
        name="aspp_context",
    )(feature_map)

    node_features = tf.keras.layers.Reshape((-1, int(region_dim)), name="region_nodes")(context_map)
    adjacency = AvaRegionSimilarityAdjacency(
        temperature=graph_temperature,
        name="region_similarity_adjacency",
    )(node_features)

    graph_context = node_features
    for block_index in range(int(graph_blocks)):
        graph_context = AvaResidualGraphConvolution(
            units=int(graph_units),
            dropout=float(graph_dropout),
            name=f"residual_graph_convolution_{block_index + 1}",
        )((graph_context, adjacency))

    region_context = tf.keras.layers.Concatenate(name="region_classification_context")([node_features, graph_context])
    region_context = tf.keras.layers.Dense(128, activation="relu", name="region_classification_dense")(region_context)
    region_context = tf.keras.layers.Dropout(float(head_dropout), name="region_classification_dropout")(region_context)
    region_logits = tf.keras.layers.Dense(1, activation="linear", dtype="float32", name="region_logits")(region_context)
    outputs = RegionLogitAggregation(
        aggregation=aggregation,
        lse_r=lse_r,
        score_activation_mode=score_activation_mode,
        name="probability",
    )(region_logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="rgnet_paper_ava_classification")


def get_rgnet_paper_ava_custom_objects() -> dict[str, object]:
    return {
        "DenseNetAvaUnitPreprocess": DenseNetAvaUnitPreprocess,
        "AvaASPPContextModule": AvaASPPContextModule,
        "AvaRegionSimilarityAdjacency": AvaRegionSimilarityAdjacency,
        "AvaResidualGraphConvolution": AvaResidualGraphConvolution,
        "RegionLogitAggregation": RegionLogitAggregation,
    }
