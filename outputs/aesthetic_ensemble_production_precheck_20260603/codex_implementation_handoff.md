# Codex Implementation Handoff: Optimized Aesthetic Ensemble

## 1. Context
We have optimized the mobile aesthetic ensemble weights using grid search on 25k AVA images and 1k AADB images. We are moving from a simple average to a calibrated weighted ensemble.

## 2. Model Configuration
The following models must be used. **Verify SHA256 checksums** on assets before proceeding.

| Model | Weight | Calibration Slope | Calibration Intercept |
| :--- | :--- | :--- | :--- |
| **NIMA** | 0.37 | 0.8126 | 1.4492 |
| **RGNet** | 0.41 | 0.0899 | 4.8656 |
| **A-LAMP** | 0.22 | 0.1637 | 4.6493 |

**ICAA Decision**: Keep ICAA as a **separate subscore**. Do NOT include it in the final holistic aesthetic score calculation.

## 3. Implementation Logic (Draft)
```dart
// 1. Get raw scores from TFLite (assumed to be scaled to 1..10 or 0..1 then * 10)
double nimaRaw = getModelScore("nima"); // Range 1..10
double rgnetRaw = getModelScore("rgnet"); // Range 1..10
double alampRaw = getModelScore("alamp"); // Range 1..10

// 2. Apply Linear Calibration
double nimaCalib = (nimaRaw * 0.8126) + 1.4492;
double rgnetCalib = (rgnetRaw * 0.0899) + 4.8656;
double alampCalib = (alampRaw * 0.1637) + 4.6493;

// 3. Holistic Ensemble
double aestheticScore = (nimaCalib * 0.37) + (rgnetCalib * 0.41) + (alampCalib * 0.22);

// 4. Clamp (Optional, following project convention)
aestheticScore = aestheticScore.clamp(1.0, 10.0);
```

## 4. Constraints
- **Do not modify technical score.**
- **Do not modify model assets.**
- **Keep ICAA separate.** If ICAA is displayed as "Color Aesthetics", keep its current scaling logic.

## 5. Files to Inspect in Flutter
- `lib/services/inference_service.dart` (or similar)
- `lib/models/aesthetic_score.dart`
- Any scoring utility or ensemble fusion logic.

## 6. Verification
- Add log statements: `[AESTHETIC_AUDIT] raw=$nimaRaw, calib=$nimaCalib, final=$aestheticScore`
- Prepare for a 100-image parity test.
