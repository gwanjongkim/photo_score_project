# Dynamic PTQ WSL Latency Report

## Scope
- This is a local WSL TensorFlow Lite interpreter measurement.
- It is not Android/iOS latency and must not be used as a production speed claim.
- Warmup runs per model: 8.
- Timed runs per model: 50.

| Model | Mean ms | Median ms | P90 ms | Max ms | Runs |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FP16 reference | 331.652110934956 | 332.08576796459965 | 336.0989319276996 | 337.78566599357873 | 50 |
| Dynamic PTQ | 191.28270630375482 | 191.37078401399776 | 194.3571365380194 | 196.38122501783073 | 50 |
