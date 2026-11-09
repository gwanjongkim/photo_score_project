"""
Repo-practical MUSIQ-style transformer model.

Based on:
- MUSIQ: Multi-scale Image Quality Transformer

Faithful parts:
- multi-scale patch token processing
- transformer encoder over patch tokens
- preserves variable aspect ratio in preprocessing before patch extraction

Approximated parts:
- uses fixed token budgets per scale instead of the paper's full sampling pipeline
- uses learned scale/position embeddings rather than the exact hash-based encoding details

Expected inputs:
- patches: [B, T, patch_size, patch_size, 3]
- positions: [B, T, 2]
- scale_ids: [B, T]
- token_mask: [B, T]

Expected outputs:
- scalar quality prediction, typically trained on KonIQ MOS
"""

from __future__ import annotations

import tensorflow as tf


def _transformer_block(
    x: tf.Tensor,
    num_heads: int,
    mlp_dim: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads,
        dropout=dropout,
        name=f"{name}_mha",
    )(norm1, norm1)
    x = tf.keras.layers.Add(name=f"{name}_attn_add")([x, attn])

    norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    mlp = tf.keras.layers.Dense(mlp_dim, activation="gelu", name=f"{name}_mlp_1")(norm2)
    mlp = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(mlp)
    mlp = tf.keras.layers.Dense(x.shape[-1], name=f"{name}_mlp_2")(mlp)
    mlp = tf.keras.layers.Dropout(dropout, name=f"{name}_drop2")(mlp)
    return tf.keras.layers.Add(name=f"{name}_mlp_add")([x, mlp])


def build_musiq_model(
    patch_size: int = 32,
    num_scales: int = 3,
    patches_per_scale: int = 16,
    embed_dim: int = 128,
    num_heads: int = 4,
    depth: int = 4,
    mlp_dim: int = 256,
    dropout: float = 0.1,
) -> tf.keras.Model:
    total_tokens = num_scales * patches_per_scale

    patches = tf.keras.Input(shape=(total_tokens, patch_size, patch_size, 3), name="patches")
    positions = tf.keras.Input(shape=(total_tokens, 2), name="positions")
    scale_ids = tf.keras.Input(shape=(total_tokens,), dtype="int32", name="scale_ids")
    token_mask = tf.keras.Input(shape=(total_tokens,), dtype="float32", name="token_mask")

    patch_encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(embed_dim, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(embed_dim),
        ],
        name="musiq_patch_encoder",
    )

    token_embeddings = tf.keras.layers.TimeDistributed(patch_encoder, name="patch_encoder_td")(patches)
    pos_embeddings = tf.keras.layers.Dense(embed_dim, activation="tanh", name="pos_embed")(positions)
    scale_embeddings = tf.keras.layers.Embedding(num_scales, embed_dim, name="scale_embed")(scale_ids)

    x = tf.keras.layers.Add(name="token_plus_meta")([token_embeddings, pos_embeddings, scale_embeddings])
    mask = tf.keras.layers.Lambda(lambda m: tf.expand_dims(m, axis=-1), name="mask_expand")(token_mask)
    x = tf.keras.layers.Multiply(name="mask_tokens")([x, mask])

    for idx in range(depth):
        x = _transformer_block(x, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, name=f"block_{idx}")

    masked_sum = tf.keras.layers.Lambda(
        lambda tensors: tf.reduce_sum(tensors[0] * tf.expand_dims(tensors[1], axis=-1), axis=1),
        name="masked_sum",
    )([x, token_mask])
    denom = tf.keras.layers.Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0),
        name="mask_denom",
    )(token_mask)
    pooled = tf.keras.layers.Lambda(lambda tensors: tensors[0] / tensors[1], name="masked_mean")([masked_sum, denom])
    pooled = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="head_ln")(pooled)
    pooled = tf.keras.layers.Dropout(dropout, name="head_drop")(pooled)
    pooled = tf.keras.layers.Dense(128, activation="gelu", name="head_dense")(pooled)
    output = tf.keras.layers.Dense(1, activation="linear", dtype="float32", name="quality")(pooled)

    return tf.keras.Model(
        inputs={
            "patches": patches,
            "positions": positions,
            "scale_ids": scale_ids,
            "token_mask": token_mask,
        },
        outputs=output,
        name="musiq_practical",
    )
