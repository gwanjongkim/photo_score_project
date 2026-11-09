# Data Inventory Notes - Unknown / Real-Mobile Fallback Audit

## Confirmed Real Mobile Rows Found
- **Found**: 50 rows in `data/debug/c6_parity_20260527/android_c6_scores.csv`. These images correspond to 2025 filenames (e.g., `20250129_102520.jpg`) and have confirmed `existing_technical_score_100` values.
- **Location**: Some images are stored in `data/Test_weight/`.

## Confirmed Unknown Rows Found
- **Not Found**: No explicit "unknown" source dataset was identified other than the generic `fallback_proxy` used in previous dry-runs.

## Fallback Proxy Status
- **Main 10k Dry-run**: Fallback proxy currently equals FLIVE image proxy.
- **Audit Update**: This audit identifies `c6_parity` and `Test_weight` as superior real-mobile proxies.

## Representative Status
- `Test_weight` and `c6_parity` images are highly representative of real A-cut mobile usage as they originate from actual device testing and product-gold labeling.
- The `existing_technical_score_100` provides the necessary production safety fallback target.
