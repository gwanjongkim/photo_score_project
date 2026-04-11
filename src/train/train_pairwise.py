from __future__ import annotations

from pathlib import Path
import argparse

import tensorflow as tf

from src.datasets.pairwise_dataset import make_pairwise_dataset
from src.models.pairwise_comparator import build_pairwise_comparator

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
    parser = argparse.ArgumentParser(description="Train a pairwise Siamese aesthetics comparator on AADB-style CSVs.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", default="score")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_pairs", type=int, default=20000)
    parser.add_argument("--similar_margin", type=float, default=0.05)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, train_pairs = make_pairwise_dataset(
        args.train_csv,
        target_col=args.target_col,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        training=True,
        max_pairs=args.max_pairs,
        similar_margin=args.similar_margin,
        random_state=42,
    )
    val_ds, val_pairs = make_pairwise_dataset(
        args.val_csv,
        target_col=args.target_col,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        training=False,
        shuffle=False,
        max_pairs=max(1000, args.max_pairs // 4),
        similar_margin=args.similar_margin,
        random_state=7,
    )

    print("Train pairs:", len(train_pairs))
    print("Val pairs:", len(val_pairs))

    model = build_pairwise_comparator(input_shape=(args.image_size, args.image_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(out_dir / "final_model.keras")
    model.export(out_dir / "saved_model")
    print("Saved:", out_dir / "final_model.keras")
    print("Exported:", out_dir / "saved_model")


if __name__ == "__main__":
    main()
