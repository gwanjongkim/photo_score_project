from pathlib import Path
import argparse
import tensorflow as tf

from src.datasets.csv_dataset import make_regression_dataset
from src.models.aesthetic_nima import build_aadb_regressor
from src.models.technical_regressor import build_technical_regressor

gpus = tf.config.list_physical_devices('GPU')
print("Visible GPUs:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("memory growth setup failed:", e)

if gpus:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("mixed precision enabled")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--task", choices=["aadb", "technical"], required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_regression_dataset(
        args.train_csv,
        target_col=args.target_col,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        training=True
    )

    val_ds = make_regression_dataset(
        args.val_csv,
        target_col=args.target_col,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        training=False,
        shuffle=False
    )

    if args.task == "aadb":
        model = build_aadb_regressor((args.image_size, args.image_size, 3))
    else:
        model = build_technical_regressor((args.image_size, args.image_size, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=["mae"]
    )

    ckpt_path = str(out_dir / "best.weights.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model.save(out_dir / "final_model.keras")
    # TFLite/Serving용 SavedModel export가 필요하면 이것도 사용
    model.export(out_dir / "saved_model")
    print("Saved:", out_dir / "final_model.keras")
    print("Exported:", out_dir / "saved_model")
if __name__ == "__main__":
    main()
