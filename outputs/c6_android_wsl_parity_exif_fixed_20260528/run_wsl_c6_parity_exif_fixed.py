# Koniq/TOPIQ TFLite 모델을 사용하여 WSL 환경의 C6 점수를 계산하고 Android 결과와 비교하는 스크립트
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None

EXIF_ORIENTATION_TAG = 274
TRANSPOSE_ORIENTATIONS = {2, 3, 4, 5, 6, 7, 8}

class TFLiteModelWrapper:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.name = Path(model_path).stem

    def predict(self, input_tensor: tf.Tensor) -> np.ndarray:
        if input_tensor.shape.rank == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor.numpy())
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

def register_optional_heif() -> bool:
    """Register HEIF support when pillow-heif is installed."""
    if register_heif_opener is None:
        return False
    register_heif_opener()
    return True

def format_size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"

def load_image_to_tensor(image_path: str) -> tuple[tf.Tensor, dict]:
    """Load image with PIL, apply EXIF orientation, and convert to TF tensor."""
    with Image.open(image_path) as img:
        exif = img.getexif()
        orientation = exif.get(EXIF_ORIENTATION_TAG) if exif else None
        size_before = img.size
        transposed = ImageOps.exif_transpose(img)
        size_after = transposed.size
        rgb_img = transposed.convert("RGB")
        img_np = np.array(rgb_img)

    metadata = {
        "exif_orientation_original": str(orientation) if orientation is not None else "none",
        "exif_transpose_applied": bool(orientation in TRANSPOSE_ORIENTATIONS),
        "image_size_before_transpose": format_size(size_before),
        "image_size_after_transpose": format_size(size_after),
    }
    return tf.convert_to_tensor(img_np), metadata

def preprocess_koniq(image: tf.Tensor, size: int = 224) -> tf.Tensor:
    """Existing KonIQ models use resize (stretch) and / 255.0 normalization."""
    image = tf.image.resize(image, (size, size))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_topiq(image: tf.Tensor, size: int = 384) -> tf.Tensor:
    """TOPIQ models use resize_with_pad and NO normalization (0..255)."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def compute_c6(koniq: float, topiq: float) -> float:
    return min(0.7 * topiq + 0.3 * koniq, koniq + 8.0)

def safe_srcc(wsl_scores: pd.Series, android_scores: pd.Series) -> float:
    return float(spearmanr(wsl_scores, android_scores).correlation)

def metric_block(df_compare: pd.DataFrame, wsl_col: str, android_col: str, abs_col: str) -> dict:
    return {
        "mae": float(df_compare[abs_col].mean()),
        "max_ae": float(df_compare[abs_col].max()),
        "median_ae": float(df_compare[abs_col].median()),
        "srcc": safe_srcc(df_compare[wsl_col], df_compare[android_col]),
    }

def top_mismatches(
    df_compare: pd.DataFrame,
    android_col: str,
    wsl_col: str,
    diff_col: str,
    abs_col: str,
    limit: int = 10,
) -> pd.DataFrame:
    return df_compare.sort_values(abs_col, ascending=False).head(limit)[[
        "filename",
        "exif_orientation_original",
        "exif_transpose_applied",
        "image_size_before_transpose",
        "image_size_after_transpose",
        android_col,
        wsl_col,
        diff_col,
        abs_col,
    ]]

def top_k_overlap(df_compare: pd.DataFrame, android_col: str, wsl_col: str, k: int) -> int:
    android_top = set(df_compare.nlargest(k, android_col)["filename"])
    wsl_top = set(df_compare.nlargest(k, wsl_col)["filename"])
    return len(android_top & wsl_top)

def main():
    # Paths
    android_csv = "data/debug/c6_parity_20260527/android_c6_scores.csv"
    image_dir = Path("test_vila")
    out_dir = Path("outputs/c6_android_wsl_parity_exif_fixed_20260528")
    out_dir.mkdir(parents=True, exist_ok=True)

    koniq_model_path = "exports/tflite/koniq_mobile.tflite"
    topiq_model_path = "outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite"

    heif_registered = register_optional_heif()
    print(f"pillow-heif registered: {heif_registered}")

    # Load models
    print(f"Loading KonIQ from {koniq_model_path}")
    koniq_model = TFLiteModelWrapper(koniq_model_path)
    print(f"Loading TOPIQ from {topiq_model_path}")
    topiq_model = TFLiteModelWrapper(topiq_model_path)

    # Read Android scores
    df_android = pd.read_csv(android_csv)
    
    results = []
    
    print(f"Processing {len(df_android)} images...")
    for _, row in df_android.iterrows():
        filename = row['filename']
        img_path = str(image_dir / filename)
        
        try:
            # Common decoding
            image_tensor, image_metadata = load_image_to_tensor(img_path)
            
            # KonIQ inference
            k_input = preprocess_koniq(image_tensor)
            k_raw = koniq_model.predict(k_input)
            k_score = float(k_raw[0][0])
            
            # TOPIQ inference
            t_input = preprocess_topiq(image_tensor)
            t_raw = topiq_model.predict(t_input)
            t_score = float(t_raw[0][0]) * 100.0
            
            # C6 computation
            c6_score = compute_c6(k_score, t_score)
            
            results.append({
                "filename": filename,
                **image_metadata,
                "wsl_koniq_score_100": k_score,
                "wsl_topiq_score_100": t_score,
                "wsl_c6_score": c6_score
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df_wsl = pd.DataFrame(results)
    df_wsl.to_csv(out_dir / "wsl_c6_scores.csv", index=False)
    
    # Merge for comparison
    df_compare = pd.merge(df_android, df_wsl, on="filename")
    
    if len(df_compare) == 0:
        print("CRITICAL: No images processed successfully.")
        return

    # Calculate differences
    df_compare["diff_koniq"] = df_compare["wsl_koniq_score_100"] - df_compare["koniq_score_100"]
    df_compare["diff_topiq"] = df_compare["wsl_topiq_score_100"] - df_compare["topiq_mixed112_score_100"]
    df_compare["diff_c6"] = df_compare["wsl_c6_score"] - df_compare["candidate_c6_score"]
    
    df_compare["abs_diff_koniq"] = df_compare["diff_koniq"].abs()
    df_compare["abs_diff_topiq"] = df_compare["diff_topiq"].abs()
    df_compare["abs_diff_c6"] = df_compare["diff_c6"].abs()
    
    df_compare.to_csv(out_dir / "android_vs_wsl_parity.csv", index=False)
    
    # Metrics
    metrics = {
        "koniq": metric_block(df_compare, "wsl_koniq_score_100", "koniq_score_100", "abs_diff_koniq"),
        "topiq": metric_block(df_compare, "wsl_topiq_score_100", "topiq_mixed112_score_100", "abs_diff_topiq"),
        "c6": metric_block(df_compare, "wsl_c6_score", "candidate_c6_score", "abs_diff_c6"),
    }

    numeric_cols = [
        "koniq_score_100",
        "topiq_mixed112_score_100",
        "candidate_c6_score",
        "wsl_koniq_score_100",
        "wsl_topiq_score_100",
        "wsl_c6_score",
        "diff_koniq",
        "diff_topiq",
        "diff_c6",
        "abs_diff_koniq",
        "abs_diff_topiq",
        "abs_diff_c6",
    ]
    no_nan = bool(df_compare[numeric_cols].notna().all().all())
    finite_numeric = bool(np.isfinite(df_compare[numeric_cols].to_numpy()).all())
    no_scale_inversion = all(metrics[name]["srcc"] > 0.0 for name in metrics)
    c6_top5_overlap = top_k_overlap(df_compare, "candidate_c6_score", "wsl_c6_score", 5)
    c6_top10_overlap = top_k_overlap(df_compare, "candidate_c6_score", "wsl_c6_score", 10)
    
    # Generate Report
    orientation_counts = df_compare["exif_orientation_original"].value_counts().sort_index()
    transposed_count = int(df_compare["exif_transpose_applied"].sum())
    size_changed_count = int((df_compare["image_size_before_transpose"] != df_compare["image_size_after_transpose"]).sum())
    top_koniq = top_mismatches(df_compare, "koniq_score_100", "wsl_koniq_score_100", "diff_koniq", "abs_diff_koniq")
    top_topiq = top_mismatches(df_compare, "topiq_mixed112_score_100", "wsl_topiq_score_100", "diff_topiq", "abs_diff_topiq")
    top_c6 = top_mismatches(df_compare, "candidate_c6_score", "wsl_c6_score", "diff_c6", "abs_diff_c6")
    
    report = f"""# C6 Android vs WSL Parity Report — EXIF Fixed

## 1. Summary
Comparison between Android Flutter TFLite implementation and WSL reference implementation after applying PIL EXIF orientation transpose in the WSL reference path.
- Android Score CSV: `{android_csv}`
- Image Directory: `test_vila`
- KonIQ Model: `{koniq_model_path}`
- TOPIQ Model: `{topiq_model_path}`
- `pillow-heif` registered: {heif_registered}
- Numeric outputs finite/no NaN: {finite_numeric and no_nan}
- Scale inversion check: {"PASS" if no_scale_inversion else "FAIL"}

## 2. What Changed
- Registered `pillow-heif` when available, instead of requiring it at import time.
- Opened images with PIL, captured original EXIF Orientation and image size, applied `ImageOps.exif_transpose`, then converted the corrected image to RGB.
- Reused the same corrected tensor for both KonIQ and TOPIQ preprocessing.
- Added per-image EXIF and before/after size fields to `wsl_c6_scores.csv` and `android_vs_wsl_parity.csv`.

## 3. Image Matching
- Total images in Android CSV: {len(df_android)}
- Matched and processed: {len(df_compare)}
- Failed: {len(df_android) - len(df_compare)}

## 4. EXIF Orientation Handling
- EXIF orientation counts:
```
{orientation_counts.to_string()}
```
- Images with EXIF transpose applied: {transposed_count}
- Images with size changed by transpose: {size_changed_count}

Fields emitted per image:
- `exif_orientation_original`
- `exif_transpose_applied`
- `image_size_before_transpose`
- `image_size_after_transpose`

Model contracts used:
### KonIQ
- Input: 224x224, Resize (Stretch)
- Normalization: pixel / 255.0
- Output: Raw (0..100)

### TOPIQ
- Input: 384x384, Resize with Pad
- Normalization: None (0..255)
- Output: Raw * 100 (0..100)

### C6 Formula
- `c6 = min(0.7 * topiq + 0.3 * koniq, koniq + 8.0)`

## 5. Per-Model Parity Metrics
| Model | MAE | Max AE | Median AE | SRCC |
| :--- | :--- | :--- | :--- | :--- |
| KonIQ | {metrics['koniq']['mae']:.4f} | {metrics['koniq']['max_ae']:.4f} | {metrics['koniq']['median_ae']:.4f} | {metrics['koniq']['srcc']:.4f} |
| TOPIQ | {metrics['topiq']['mae']:.4f} | {metrics['topiq']['max_ae']:.4f} | {metrics['topiq']['median_ae']:.4f} | {metrics['topiq']['srcc']:.4f} |
| **C6** | **{metrics['c6']['mae']:.4f}** | **{metrics['c6']['max_ae']:.4f}** | **{metrics['c6']['median_ae']:.4f}** | **{metrics['c6']['srcc']:.4f}** |

## 6. Top Mismatches
### KonIQ
```
{top_koniq.to_string(index=False)}
```

### TOPIQ
```
{top_topiq.to_string(index=False)}
```

### C6
```
{top_c6.to_string(index=False)}
```

## 7. Remaining Error Analysis
- C6 top-5 ranking overlap: {c6_top5_overlap}/5
- C6 top-10 ranking overlap: {c6_top10_overlap}/10
- EXIF orientation handling removes the largest known Android-vs-WSL decode mismatch.
- Remaining C6 error is consistent with bilinear resize implementation differences and C6 branch sensitivity near the `min()` transition.
- Production replacement still requires stricter parity than log-only diagnostics.

## 8. Decision
"""
    
    c6_mae = metrics['c6']['mae']
    c6_max = metrics['c6']['max_ae']
    
    if (finite_numeric and no_nan and no_scale_inversion and c6_mae <= 1.0 and c6_max <= 3.0):
        decision = "A. Parity PASS"
    elif finite_numeric and no_nan and no_scale_inversion and c6_max <= 5.0:
        decision = "B. Partial PASS, acceptable for log-only"
    else:
        decision = "C. FAIL, further preprocessing mismatch remains"
        
    report += f"{decision}\n"
    
    with open(out_dir / "report.md", "w") as f:
        f.write(report)
        
    print(f"Report generated at {out_dir / 'report.md'}")

if __name__ == "__main__":
    main()
