# Checklist

- [x] Create isolated source-dimension-matched diagnostic script.
- [x] Register `pillow_heif`.
- [x] Apply EXIF transpose before RGB conversion.
- [x] Resize source to Android logged decoded dimensions.
- [x] Apply PIL bilinear resize-with-pad to `384x384`.
- [x] Compute required tensor fingerprints and comparison fields.
- [x] Compare against previous `pil_bilinear_resize_with_pad`.
- [x] Write `source_dimension_matched_summary.csv`.
- [x] Write `per_image_source_dimension_matched_comparison.csv`.
- [x] Write `report.md`.
- [x] Print report path and key terminal metrics.
