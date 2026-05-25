# RGNet-v1-Hybrid-Spatial Context Notes

- This is a paper-oriented enhancement, not a full RGNet paper reproduction.
- The spatial/geometric Gaussian prior is an experimental addition on top of the existing semantic feature-similarity adjacency.
- The existing semantic adjacency must remain the default for old configs and checkpoints.
- The requested smallrun output directory is also used for these run notes.
- `adjacency_type=semantic` keeps the existing `V1RegionSimilarityAdjacency` path.
- `adjacency_type=hybrid_spatial` uses the existing semantic adjacency plus a row-normalized Gaussian prior over the post-ASPP spatial grid.
- The sandboxed default smoke could not initialize CUDA and fell back to random DenseNet weights. The escalated hybrid smoke used the RTX 4070 SUPER and cached DenseNet weights.
- The escalated hybrid smoke completed but spent significant time in TensorFlow/XLA GPU compilation, so the 1024/256 smallrun must be treated as a bounded stability run rather than a quick smoke.
- The 1024/256 hybrid smallrun completed 5 epochs. Best validation loss was epoch 3 with val_mae about 0.1269.
- Bounded best-weight evaluation produced val SRCC about 0.5476 and PLCC about 0.5520 on 256 validation samples, plus test-subset SRCC about 0.5072 and PLCC about 0.5285 on 256 test samples.
- This is stable enough to run a full AADB experiment, but the bounded metrics do not prove improvement over the current full AADB SRCC baseline of 0.6819.
