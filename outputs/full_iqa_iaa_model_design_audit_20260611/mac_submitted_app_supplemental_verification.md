# Mac Submitted App Supplemental Verification

## 1. Purpose
The purpose of this supplemental verification note is to reconcile the `FAILED_WEIGHT_MISMATCH` result reported in `codex_final_verification.md` with the actual implementation in the submitted Flutter app. This note documents the manual verification performed on the Mac build environment.

## 2. Why Codex Verification Failed
The automated Codex verification reported a weight mismatch because it could not access the actual submitted project path on the Mac. Instead, it inspected a WSL-accessible Windows checkout at:
`/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app`

This Windows checkout contained outdated experimental weights (NIMA 0.10 / RGNet 0.50 / A-LAMP 0.40) and different active model contracts (GPU variants), which do not reflect the final submitted version.

## 3. Exact Submitted App Path
The actual submitted app, verified via Mac terminal, is located at:
`/Users/gwanjung_mac/StudioProjects/pozy_app`

## 4. Manual Mac Verification Evidence
The following results were confirmed via the Mac terminal at the submitted app path:

**Aesthetic Weights (`aesthetic_ensemble_weights.dart`):**
```text
grep -n "_defaultNimaWeight|_defaultRgnetWeight|_defaultAlampWeight" lib/feature/a_cut/model/aesthetic_ensemble_weights.dart
2:  static const double _defaultNimaWeight = 0.40;
3:  static const double _defaultRgnetWeight = 0.45;
4:  static const double _defaultAlampWeight = 0.15;
```

**Assets Declaration (`pubspec.yaml`):**
```text
grep -n "assets/models" pubspec.yaml
61:    - assets/models/
```

**Model Contracts (`aesthetic_model_contract.dart`):**
```text
sed -n '527,540p' lib/feature/a_cut/layer/inference/aesthetic_model_contract.dart
const List<AestheticModelContract> defaultTechnicalModelContracts = [
  koniqMobileContract,
  fliveImageMobileContract,
];

const List<AestheticModelContract> activeAestheticEnsembleContracts = [
  nimaMobileContract,
  mobileAlampV2Contract,
  rgnetPilResizeAadbContract,
];
```

**Experimental Features (`experimental_features.dart`):**
```text
sed -n '50,58p' lib/config/experimental_features.dart
static const bool enableTopiqFlive6040Shadow = bool.fromEnvironment(
  'POZY_ENABLE_TOPIQ_FLIVE_60_40_SHADOW',
  defaultValue: false,
);
```

## 5. Submitted App Model Configuration
*   **Technical IQA Stack:** KonIQ + FLIVE
*   **Aesthetic IAA Stack:** NIMA + Mobile A-LAMP v2 + RGNet
*   **Default Aesthetic Weights:** NIMA 0.40 / RGNet 0.45 / A-LAMP 0.15
*   **TOPIQ Shadow Flag:** Default `false`

## 6. Comparison Against Main Audit Report
The manual evidence confirms that the main audit report (`full_model_design_audit.md`) correctly describes the submitted implementation.
*   **Production Models:** Identity matched.
*   **Weights:** The report accurately documents the submitted defaults (0.40 / 0.45 / 0.15) while referencing the offline search optimum (0.37 / 0.41 / 0.22) as research context.
*   **Contract Logic:** Identity matched.

## 7. Final Interpretation
The `FAILED_WEIGHT_MISMATCH` result in the Codex report is a **path-scope mismatch** and does not constitute evidence of a contradiction between the audit report and the submitted Mac app. Manual terminal verification conclusively proves that the main audit report is aligned with the submitted source code.

## 8. Remaining Caveat
For a fully automated end-to-end verification, the Mac submitted checkout must be mounted to WSL or verified by running Codex directly on the Mac filesystem. However, the provided terminal evidence is sufficient for the accuracy of the current IQA/IAA model design audit.
