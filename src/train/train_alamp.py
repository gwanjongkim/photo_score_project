from __future__ import annotations

from pathlib import Path
import argparse

import tensorflow as tf

from src.datasets.native_size_dataset import make_alamp_dataset
from src.models.alamp import build_alamp_model

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a practical A-Lamp-style aesthetic model.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="score")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--global_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--num_patches", type=int, default=5)
    parser.add_argument("--patch_scale", type=float, default=0.35)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_alamp_dataset(
        args.train_csv,
        target_col=args.target_col,
        batch_size=args.batch_size,
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        patch_scale=args.patch_scale,
        training=True,
    )
    val_ds = make_alamp_dataset(
        args.val_csv,
        target_col=args.target_col,
        batch_size=args.batch_size,
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        patch_scale=args.patch_scale,
        training=False,
        shuffle=False,
    )

    model = build_alamp_model(
        global_size=args.global_size,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="mse",
        metrics=["mae"],
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
