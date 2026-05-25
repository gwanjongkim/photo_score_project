# RGNet-v1-Hybrid-Spatial Plan

1. Inspect the current RGNet paper-v1 model, trainer, and mean-aggregation config so the semantic default remains unchanged.
2. Add an opt-in hybrid semantic-spatial adjacency path with `adjacency_type=hybrid_spatial`, `spatial_alpha`, and `spatial_sigma`.
3. Record adjacency settings, node count, and model parameter counts in the training summary.
4. Verify with `py_compile`, a semantic default smoke run, a hybrid smoke run, and a 1024/256 hybrid smallrun if smoke passes.

