"""A-LAMP-paper-MPNet model.

A New-MP-Net-style Multi-Patch subnet implementation using a shared VGG16 backbone
and orderless statistical aggregation (Mean + Max pooling).
"""

from __future__ import annotations
import tensorflow as tf

MODEL_VARIANT = "A-LAMP-paper-MPNet"
MODEL_DESCRIPTION = "New-MP-Net-style Multi-Patch subnet"

@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class PatchReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):
        # [B, 5, H, W, 3] -> [B*5, H, W, 3]
        return tf.reshape(inputs, [-1, self.patch_size, self.patch_size, 3])

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class PatchAggregationLayer(tf.keras.layers.Layer):
    def __init__(self, patch_count: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_count = patch_count

    def call(self, inputs):
        # [B*5, D] -> [B, 5, D]
        d = inputs.shape[-1]
        x = tf.reshape(inputs, [-1, self.patch_count, d])
        
        # Aggregation: mean + max
        mean_pool = tf.reduce_mean(x, axis=1)
        max_pool = tf.reduce_max(x, axis=1)
        return tf.concat([mean_pool, max_pool], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_count": self.patch_count})
        return config

@tf.keras.utils.register_keras_serializable(package="photo_score_project")
class VggPreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Assuming input is [0, 1], scale to [0, 255] and apply VGG16 preprocess
        x = inputs * 255.0
        return tf.keras.applications.vgg16.preprocess_input(x)

def build_alamp_paper_mpnet_model(
    patch_count: int = 5,
    patch_size: int = 224,
    backbone_weights: str | None = "imagenet",
    feature_dim: int = 256,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Builds a Multi-Patch subnet model using shared VGG16."""
    
    # Shared backbone
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=backbone_weights,
        input_shape=(patch_size, patch_size, 3),
        pooling="avg",
        name="vgg16_backbone",
    )
    base_model.trainable = True
    
    # Multi-patch input: [B, 5, 224, 224, 3]
    patches_input = tf.keras.Input(shape=(patch_count, patch_size, patch_size, 3), name="patches")
    
    # Reshape for shared backbone
    x = PatchReshapeLayer(patch_size=patch_size)(patches_input)
    
    # VGG16 preprocessing
    x = VggPreprocessLayer()(x)
    
    # Pass through shared backbone
    patch_features = base_model(x)
    
    # Reshape back and aggregate
    fused = PatchAggregationLayer(patch_count=patch_count)(patch_features)
    
    # Prediction head
    fused = tf.keras.layers.Dense(feature_dim, activation="relu", name="fused_dense")(fused)
    fused = tf.keras.layers.Dropout(dropout_rate, name="dropout")(fused)
    output = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(fused)
    
    model = tf.keras.Model(inputs=patches_input, outputs=output, name="alamp_paper_mpnet")
    return model

def get_alamp_paper_mpnet_custom_objects() -> dict[str, object]:
    return {
        "PatchReshapeLayer": PatchReshapeLayer,
        "PatchAggregationLayer": PatchAggregationLayer,
        "VggPreprocessLayer": VggPreprocessLayer,
    }
