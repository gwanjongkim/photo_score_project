# RGNet-v1-Cascaded-DenseASPP Plan

1. Keep existing RGNet paper-v1 parallel ASPP behavior as the default.
2. Add an opt-in cascaded DenseASPP context module with four dense atrous layers using rates 3, 6, 12, and 18.
3. Thread context module settings through training and weights-only evaluation.
4. Verify with compile checks, default compatibility smoke, DenseASPP smoke, and a bounded 1024/256 smallrun if smoke passes.

