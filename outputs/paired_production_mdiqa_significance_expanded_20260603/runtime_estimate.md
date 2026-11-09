# Runtime Estimate for Expanded Production Predictions

## 1. Measured Performance (GPU)
- **Environment**: `.venv_tf` (GPU enabled)
- **Average Time per Image**: 3.5 seconds (steady state)
- **Batching**: Not implemented (serial image-by-image)
- **Overhead**: 9+ models per image (KonIQ, FLIVE, AADB, MUSIQ)

## 2. Projections
| Dataset | Count (n) | Est. Runtime (mins) |
|---|---|---|
| KonIQ-10k (Sample) | 100 | 5.8 |
| KonIQ-10k (Sample) | 500 | 29.2 |
| KonIQ-10k (Sample) | 1000 | 58.3 |
| SPAQ (Sample) | 100 | 5.8 |
| SPAQ (Full) | 1125 | 65.6 |
| **Total (n=1000 + 1125)** | **2125** | **123.9** |

## 3. Approved Expansion (under 60 min limit)
- **KonIQ-10k**: n=500
- **SPAQ**: n=500
- **Total**: n=1000
- **Estimated Time**: ~58.3 minutes

## 4. Request for Approval (Higher N)
To run all SPAQ (1125) and more KonIQ (1000+), an estimated **2+ hours** is required. 
Current plan: Run **n=500 per dataset** to stay within safety limits.
