# Plan

Implement the recipe-adjusted A-LAMP Multi-Patch teacher paper-capacity 4096 experiment behind opt-in flags.

1. Add optional L2 regularization to Dense kernels/biases with defaults at `0.0`.
2. Add configurable shared flatten-dense projection depth while keeping the default at one layer.
3. Add train-only consistent horizontal flip over all five cropped patches before VGG16 preprocessing.
4. Add optimizer selection only where it fits cleanly, preserving Adam as the default.
5. Validate with compile checks, default smoke, recipe smoke, then the requested 8192/2048 midrun only if smoke succeeds.
