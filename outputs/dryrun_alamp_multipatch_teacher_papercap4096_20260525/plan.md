# Plan

Implement the A-LAMP Multi-Patch teacher paper-capacity variant behind explicit capacity flags while preserving the current GAP baseline defaults.

1. Add model options for GAP vs Flatten+Dense patch projection and configurable repeated dense head layers.
2. Keep default behavior equivalent to the current GAP path: GAP patch features, 512 implied patch feature size, one 256-unit head layer, dropout 0.5.
3. Add trainer CLI/config resolution and summary metadata for the capacity settings.
4. Run compile validation, then only the requested default compatibility smoke and paper-capacity 4096 smoke.
