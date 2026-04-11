from __future__ import annotations

from pathlib import Path
import argparse

import tensorflow as tf
# Disable XLA JIT globally for MUSIQ before any other TF calls to avoid dynamic shape compilation errors
tf.config.optimizer.set_jit(False)

from src.datasets.native_size_dataset import make_musiq_dataset
from src.models.musiq import build_musiq_model

gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        print("memory growth setup failed:", exc)

if gpus:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("mixed precision enabled")
    # Disable XLA JIT for MUSIQ due to dynamic tensor shapes in transformer layers
    tf.config.optimizer.set_jit(False)


def _parse_scale_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a repo-practical MUSIQ-style transformer model.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="mos")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--scale_sizes", default="224,384,512")
    parser.add_argument("--patches_per_scale", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=256)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale_sizes = _parse_scale_sizes(args.scale_sizes)
    train_ds = make_musiq_dataset(
        args.train_csv,
        target_col=args.target_col,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        scale_sizes=scale_sizes,
        patches_per_scale=args.patches_per_scale,
        training=True,
    )
    val_ds = make_musiq_dataset(
        args.val_csv,
        target_col=args.target_col,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        scale_sizes=scale_sizes,
        patches_per_scale=args.patches_per_scale,
        training=False,
        shuffle=False,
    )

    model = build_musiq_model(
        patch_size=args.patch_size,
        num_scales=len(scale_sizes),
        patches_per_scale=args.patches_per_scale,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="mse",
        metrics=["mae"],
        jit_compile=False,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(out_dir / "final_model.keras")
    model.export(out_dir / "saved_model")
    print("Saved:", out_dir / "final_model.keras")
    print("Exported:", out_dir / "saved_model")


if __name__ == "__main__":
    main()
