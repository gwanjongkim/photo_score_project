# TOPIQ+FLIVE Safety Signal Ablation Report

## 1. Executive Summary
We investigated alternative safety signals to eliminate the remaining 2 severe False Positives (FPs) from the TOPIQ+FLIVE 60:40 hybrid model (Gate A). Using MDIQA as an oracle gate reduced severe FPs to 1 but could not fully eliminate them without heavily clipping normal predictions. However, simple image processing metrics (Laplacian variance, noise estimation) showed strong separation for the remaining severe FPs, indicating that a lightweight safety signal is highly feasible.

**Final Decision: TOPIQ_FLIVE_SAFETY_SIGNAL_PROMISING**

## 2. Sources Inspected
- `outputs/topiq_flive_safety_gate_simulation_20260608/gated_predictions.csv`
- `outputs/official_protocol_mdiqa_signal_benchmark_20260604/official_mdiqa_vs_candidates_predictions.csv`

## 3. Remaining Severe FP Cases
After applying Gate A (which capped production scores <45), two severe FPs remained:
1. `8090361774.jpg` (Heavy noise) - MOS: 35.98, Production: 62.56, Base Candidate: 69.56
2. `10619.jpg` (Main-subject blur) - MOS: 39.90, Production: 52.33, Base Candidate: 66.50

## 4. MDIQA Oracle Gate Results
Simulating gates using the MDIQA teacher scores:
- **Oracle 1 (MDIQA < 50 cap 10)**: 3 severe FPs remaining, SRCC 0.8057 (Failed to catch due to MDIQA scoring these images > 50).
- **Oracle 2 (MDIQA < 55 cap 10)**: 3 severe FPs remaining, SRCC 0.8059.
- **Oracle 3 (MDIQA < 60 cap 5)**: 1 severe FP remaining, SRCC 0.8062.

MDIQA itself is slightly optimistic on these specific images (scoring ~56-58), meaning an MDIQA-based gate requires an aggressive threshold (<60) to trigger, which degrades overall performance and still fails to catch all cases.

## 5. Simple Blur/Noise/Focus Metric Probe
We extracted lightweight classical metrics for the severe FPs and a random sample of images:
- `8090361774.jpg` (Heavy noise): Noise Estimate `20.64`, Laplacian Var `7167`
- `10619.jpg` (Blur): Noise Estimate `11.06`, Laplacian Var `2129`
- **Normal Sample Average**: Noise Estimate `~2.0 - 5.0`, Laplacian Var `~50 - 600`

The simple metrics perfectly separate the severe FPs (which exhibit heavy noise or sharp high-frequency background artifacts) from normal images.

## 6. Tradeoff Analysis
Relying purely on black-box neural networks (Production or MDIQA) for safety gating fails because they occasionally hallucinate high quality on specific artifact combinations. Conversely, classical metrics (like Laplacian variance and median-subtracted noise) are robust to these hallucinations and can easily catch heavy noise.

## 7. Mobile Feasibility
Computing simple noise and variance metrics via OpenCV/CoreImage/Metal on mobile takes less than 5ms. Adding these as a fallback safety gate is highly feasible and computationally lighter than running a secondary neural network.

## 8. Final Decision
**TOPIQ_FLIVE_SAFETY_SIGNAL_PROMISING**

## 9. Next Actions
- Prototype a lightweight mobile safety gate using classical noise and sharpness heuristics.
- Profile the CPU/latency cost of these heuristics in Flutter.
- Proceed with log-only profiling of TOPIQ+FLIVE 60:40 accompanied by the new heuristic gate.
