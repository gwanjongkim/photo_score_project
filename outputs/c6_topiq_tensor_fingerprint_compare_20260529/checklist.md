# Checklist

- [x] Create isolated fingerprint comparison script.
- [x] Register `pillow_heif` and apply EXIF transpose before RGB conversion.
- [x] Implement all four WSL backend candidates.
- [x] Compute required tensor fingerprints for every matched image/backend.
- [x] Compare Android vs WSL checksum, stats, channel means/stds, patch means, first/last values, and geometry.
- [x] Join C6 mismatch context from prior backend ablation.
- [x] Write `backend_fingerprint_summary.csv`.
- [x] Write `per_image_fingerprint_comparison.csv`.
- [x] Write `report.md`.
- [x] Record confirmed facts separately from assumptions.
