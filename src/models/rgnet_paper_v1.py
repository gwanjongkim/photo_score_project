# RGNet 논문 지향 v1 AADB 회귀 모델을 정의한다.
"""RGNet-paper-v1 approximation for isolated AADB regression experiments.

This is not an official paper reproduction. It is a paper-oriented improvement
over ``src/models/rgnet_paper.py`` v0, while leaving v0 reproducible.

v1 additions:
    - ASPP multi-scale context after the DenseNet121 feature map.
    - Three residual graph-convolution blocks by default.
    - Region-level score prediction.
    - Configurable score aggregation, defaulting to Log-Sum-Exp with r=4.

Aggregation choice:
    Region scores are sigmoid-normalized first, then aggregated. This keeps each
    node score in the AADB target range and makes mean/max/LSE outputs remain in
    [0, 1]. LSE uses (1 / r) * log(mean(exp(r * scores))) with a numerically
    stable max-subtraction implementation.
"""

from __future__ import annotations

import math
import sys

import tensorflow as tf

MODEL_VARIANT = "RGNet-paper-v1 approximation"


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
class DenseNetV1UnitPreprocess(tf.keras.layers.Layer):
    """Converts [0, 1] RGB tensors to DenseNet121 ImageNet preprocessing."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)
        return tf.keras.applications.densenet.preprocess_input(inputs * 255.0)

    def get_config(self) -> dict[str, object]:
        return dict(super().get_config())


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class ASPPContextModule(tf.keras.layers.Layer):
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
class V1RegionSimilarityAdjacency(tf.keras.layers.Layer):
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
class V1HybridSpatialAdjacency(tf.keras.layers.Layer):
    """Mixes semantic cosine adjacency with a normalized spatial Gaussian prior."""

    def __init__(
        self,
        temperature: float = 0.25,
        alpha: float = 0.0,
        sigma: float = 0.25,
        epsilon: float = 1e-6,
        spatial_height: int | None = None,
        spatial_width: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.spatial_height = None if spatial_height is None else int(spatial_height)
        self.spatial_width = None if spatial_width is None else int(spatial_width)
        self.semantic_adjacency = V1RegionSimilarityAdjacency(temperature=self.temperature)
        self._spatial_adjacency: tf.Tensor | None = None

    def build(self, input_shape):
        node_count = input_shape[1]
        if self.spatial_height is None or self.spatial_width is None:
            if node_count is None:
                raise ValueError("V1HybridSpatialAdjacency needs a static node count or spatial_height/spatial_width.")
            node_count = int(node_count)
            side = int(round(math.sqrt(node_count)))
            if side * side != node_count:
                raise ValueError(
                    "V1HybridSpatialAdjacency can infer only square grids; "
                    f"got node_count={node_count}."
                )
            self.spatial_height = side
            self.spatial_width = side
        self._spatial_adjacency = self._make_spatial_adjacency(self.spatial_height, self.spatial_width)
        super().build(input_shape)

    def _make_spatial_adjacency(self, height: int, width: int) -> tf.Tensor:
        y = tf.linspace(0.0, 1.0, int(height))
        x = tf.linspace(0.0, 1.0, int(width))
        grid_y, grid_x = tf.meshgrid(y, x, indexing="ij")
        coords = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], axis=-1)
        diff = coords[:, tf.newaxis, :] - coords[tf.newaxis, :, :]
        dist2 = tf.reduce_sum(tf.square(diff), axis=-1)
        sigma = tf.maximum(tf.cast(self.sigma, tf.float32), tf.constant(self.epsilon, tf.float32))
        spatial = tf.exp(-dist2 / (2.0 * tf.square(sigma)))
        row_sums = tf.reduce_sum(spatial, axis=-1, keepdims=True)
        return spatial / tf.maximum(row_sums, tf.cast(self.epsilon, tf.float32))

    def call(self, node_features: tf.Tensor) -> tf.Tensor:
        semantic = self.semantic_adjacency(node_features)
        if self._spatial_adjacency is None:
            raise RuntimeError("V1HybridSpatialAdjacency was called before build.")
        spatial = tf.cast(self._spatial_adjacency, semantic.dtype)
        spatial = tf.broadcast_to(spatial[tf.newaxis, :, :], tf.shape(semantic))
        alpha = tf.clip_by_value(tf.cast(self.alpha, semantic.dtype), 0.0, 1.0)
        mixed = ((1.0 - alpha) * semantic) + (alpha * spatial)
        row_sums = tf.reduce_sum(mixed, axis=-1, keepdims=True)
        return mixed / tf.maximum(row_sums, tf.cast(self.epsilon, mixed.dtype))

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "temperature": self.temperature,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "spatial_height": self.spatial_height,
            "spatial_width": self.spatial_width,
        }


@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class V1ResidualGraphConvolution(tf.keras.layers.Layer):
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
class RegionScoreAggregation(tf.keras.layers.Layer):
    """Aggregates per-region scores into one image-level score."""

    def __init__(self, aggregation: str = "lse", lse_r: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        aggregation = str(aggregation).lower()
        if aggregation not in {"mean", "max", "lse"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        self.aggregation = aggregation
        self.lse_r = float(lse_r)

    def call(self, region_scores: tf.Tensor) -> tf.Tensor:
        region_scores = tf.cast(region_scores, tf.float32)
        if self.aggregation == "mean":
            return tf.reduce_mean(region_scores, axis=1)
        if self.aggregation == "max":
            return tf.reduce_max(region_scores, axis=1)

        r = tf.maximum(tf.cast(self.lse_r, region_scores.dtype), tf.constant(1e-6, region_scores.dtype))
        scaled = r * region_scores
        max_scaled = tf.reduce_max(scaled, axis=1, keepdims=True)
        stable_mean = tf.reduce_mean(tf.exp(scaled - max_scaled), axis=1, keepdims=True)
        lse = (tf.math.log(tf.maximum(stable_mean, 1e-12)) + max_scaled) / r
        return tf.squeeze(lse, axis=1)

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "aggregation": self.aggregation, "lse_r": self.lse_r}


def build_rgnet_paper_v1_model(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    backbone_weights: str | None = "imagenet",
    region_dim: int = 256,
    graph_units: int = 256,
    graph_blocks: int = 3,
    graph_temperature: float = 0.25,
    graph_dropout: float = 0.1,
    head_dropout: float = 0.3,
    dilation_rates: tuple[int, ...] = (1, 3, 6, 12, 18),
    aggregation: str = "lse",
    lse_r: float = 4.0,
    adjacency_type: str = "semantic",
    spatial_alpha: float = 0.0,
    spatial_sigma: float = 0.25,
) -> tf.keras.Model:
    """Builds RGNet-paper-v1 approximation for AADB regression."""

    adjacency_type = str(adjacency_type).lower()
    if adjacency_type not in {"semantic", "hybrid_spatial"}:
        raise ValueError(f"Unsupported adjacency_type: {adjacency_type}")

    backbone = _build_densenet121_backbone(input_shape=input_shape, weights=backbone_weights)
    backbone.trainable = True

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = DenseNetV1UnitPreprocess(name="densenet_preprocess")(inputs)
    feature_map = backbone(x, training=False)
    context_map = ASPPContextModule(
        filters=int(region_dim),
        dilation_rates=dilation_rates,
        name="aspp_context",
    )(feature_map)

    node_features = tf.keras.layers.Reshape((-1, int(region_dim)), name="region_nodes")(context_map)
    spatial_height = context_map.shape[1]
    spatial_width = context_map.shape[2]
    if adjacency_type == "hybrid_spatial":
        adjacency = V1HybridSpatialAdjacency(
            temperature=graph_temperature,
            alpha=spatial_alpha,
            sigma=spatial_sigma,
            spatial_height=None if spatial_height is None else int(spatial_height),
            spatial_width=None if spatial_width is None else int(spatial_width),
            name="hybrid_spatial_adjacency",
        )(node_features)
    else:
        adjacency = V1RegionSimilarityAdjacency(
            temperature=graph_temperature,
            name="region_similarity_adjacency",
        )(node_features)

    graph_context = node_features
    for block_index in range(int(graph_blocks)):
        graph_context = V1ResidualGraphConvolution(
            units=int(graph_units),
            dropout=float(graph_dropout),
            name=f"residual_graph_convolution_{block_index + 1}",
        )((graph_context, adjacency))

    region_context = tf.keras.layers.Concatenate(name="region_score_context")([node_features, graph_context])
    region_context = tf.keras.layers.Dense(128, activation="relu", name="region_score_dense")(region_context)
    region_context = tf.keras.layers.Dropout(float(head_dropout), name="region_score_dropout")(region_context)
    region_scores = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="region_scores")(region_context)
    outputs = RegionScoreAggregation(
        aggregation=aggregation,
        lse_r=lse_r,
        name="score",
    )(region_scores)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rgnet_paper_v1_aadb_regression")
    feature_map_node_count = None
    if spatial_height is not None and spatial_width is not None:
        feature_map_node_count = int(spatial_height) * int(spatial_width)
    model._rgnet_paper_v1_config = {
        "adjacency_type": adjacency_type,
        "spatial_alpha": float(spatial_alpha),
        "spatial_sigma": float(spatial_sigma),
        "feature_map_height": None if spatial_height is None else int(spatial_height),
        "feature_map_width": None if spatial_width is None else int(spatial_width),
        "feature_map_node_count": feature_map_node_count,
    }
    return model


def get_rgnet_paper_v1_custom_objects() -> dict[str, object]:
    return {
        "DenseNetV1UnitPreprocess": DenseNetV1UnitPreprocess,
        "ASPPContextModule": ASPPContextModule,
        "V1RegionSimilarityAdjacency": V1RegionSimilarityAdjacency,
        "V1HybridSpatialAdjacency": V1HybridSpatialAdjacency,
        "V1ResidualGraphConvolution": V1ResidualGraphConvolution,
        "RegionScoreAggregation": RegionScoreAggregation,
    }
