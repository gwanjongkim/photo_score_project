# Next Action: Deeper Audit for Missing Splits

## Final Recommendation
Proceed with **MUSIQ protocol reproduction** for KonIQ-10k and SPAQ immediately, as all required files are present. For PaQ-2-PiQ, perform a targeted search for the official 30k/7.7k/1.8k split before falling back to a custom 70/10/20 split.

## Exact Next Codex Task
"Locate or reconstruct the official PaQ-2-PiQ 30k/7.7k/1.8k split. Scan all scripts and subdirectories for any mention of the official partition or a mapping file that identifies the 1,800 test images."

## Files/Folders to Inspect
- `data/raw/paq2piq/` (Re-scan for any hidden or misnamed files)
- `scripts/` (Check for any data preparation scripts that might have hardcoded logic for this split)
- `external/` (Check if other external repositories contain the split file)

## Warning
**Do NOT invent a custom benchmark.** If the official split cannot be found, explicitly label the fallback as "NR-IQA 70/10/20 (Custom Split)" and state that it deviates from the MUSIQ/PaQ-2-PiQ paper protocols.
