# Revision Changelog — Full IQA/IAA Model Design Audit

_Date: 2026-06-13. Scope: Documentation revision based on Mac Flutter app asset evidence._

## Files Edited
- `outputs/full_iqa_iaa_model_design_audit_20260611/full_model_design_audit.md`
- `outputs/full_iqa_iaa_model_design_audit_20260611/technical_iqa_assessment.csv`
- `outputs/full_iqa_iaa_model_design_audit_20260611/aesthetic_iaa_assessment.csv`
- `outputs/full_iqa_iaa_model_design_audit_20260611/mobile_deployment_risk_review.csv`
- `outputs/full_iqa_iaa_model_design_audit_20260611/production_recommendation.md`
- `outputs/full_iqa_iaa_model_design_audit_20260611/thesis_writing_notes.md`

## Summary of Changes

### 1. Mac Flutter App Asset Evidence Integration
- Added **"Current Flutter App Asset Snapshot"** section to the main audit report.
- Confirmed that production assets (**KonIQ/FLIVE** and **NIMA/RGNet/A-LAMP**) are present in `/Users/gwanjung_mac/StudioProjects/pozy_app/assets/models`.
- Reclassified **TOPIQ** as a **shadow/log-only** candidate (assets present but not in the production ensemble).
- Confirmed **ICAA** and **MDIQA student** models are **absent** from the production asset folder.
- Listed non-ensemble assets (FastSCNN, Lighting, YOLO) present in the app.

### 2. Wording and Labeling
- Replaced all instances of **"gold-standard"** for the MDIQA teacher with **"strong reference teacher"** or **"performance-ceiling reference"** (Korean: **강력한 연구용 기준 teacher / 성능 상한 참고 모델**).
- Corrected status descriptions in CSV files to match the confirmed asset listing.

### 3. Timing and Latency Caveats
- Added caveats to model timing numbers (NIMA ~177ms, etc.) stating they are based on **representative Android/project logs**.
- Clarified that end-to-end app latency includes image loading and preprocessing (e.g., A-LAMP's CPU cost).
- Noted that a single consolidated timing report was not located.

### 4. Cleanup
- Removed trailing generated tags (`</content>`, `</invoke>`) from the end of files.

## Status of Production Recommendation
- **Unchanged**: The core recommendation remains `FULL_MODEL_DESIGN_AUDIT_SUPPORTS_CURRENT_KONIQ_FLIVE_AND_NIMA_RGNET_ALAMP_PRODUCTION`.

## Remaining Required Checks
- Verify `pubspec.yaml` asset declarations in the Flutter project (Completed: assets/models/ declared).
- Verify POZY_ENABLE_TOPIQ_FLIVE_60_40_SHADOW feature flag default value (expected `false`).
- Confirm no active production paths for ICAA or MDIQA student models in the Dart evaluation contract (Completed: ICAA/MDIQA excluded from active ensemble).

## Submitted Flutter App Alignment (2026-06-13 update)
- Documented actual submitted weights: NIMA 0.40 / RGNet 0.45 / A-LAMP 0.15.
- Separated offline weight-search recommendation (0.37 / 0.41 / 0.22) from submitted defaults.
- Confirmed no app code changes recommended; the audit now evaluates implementation as-is.
- Refined TOPIQ status as shadow/log-only, not part of default technical score.
