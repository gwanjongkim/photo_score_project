# Next Evaluation Recommendation

The bounded evaluation (300 rows) successfully verified that the evaluation pipeline is stable and the baseline models (`koniq_standalone` and `flive_standalone`) are producing valid scores. The performance metrics are consistent with paper protocols, and the runtime of ~7.5ms per prediction ensures that a larger evaluation is feasible without significant overhead.

## Exact Next Evaluation Size
**Full Test Split Evaluation (Image-level only)**
- KonIQ-10k: Run_01 Test (2,015 images)
- SPAQ: Official Test (1,125 images)
- FLIVE: Image-test (1,827 images)
- **Total: 4,967 images**

## Reason
Evaluating the full test splits will provide statistically robust metrics for the research-protocol baseline report. The bounded run already confirmed that there are no technical blockers (e.g., file path errors or scale issues that would require a "C" action). Proceeding directly to full test sets is the most efficient next step.

## Stop Conditions
- Total execution time exceeds 5 minutes.
- Memory usage spikes significantly during batch processing.
- Any image loading failure rate exceeding 1% (approx. 50 images).
