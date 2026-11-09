# Mac-side Model Asset Verification Commands

Run these commands on your Mac to verify that the Flutter assets match the audited WSL artifacts.

## 1. List assets
```bash
ls -lh /Users/gwanjung_mac/StudioProjects/pozy_app/assets/models/
```

## 2. Compute Checksums
```bash
shasum -a 256 /Users/gwanjung_mac/StudioProjects/pozy_app/assets/models/*.tflite
```

## 3. Expected Values (WSL Audited)
| Filename | Expected SHA256 |
| :--- | :--- |
| **nima_mobile_fixed_preproc_fp16.tflite** | `ccbb51123f8414fdb40e2ba6a9c4a3b7be11cbf00e0bd14442923da0de46d255` |
| **rgnet_pil_resize_aadb_fp16.tflite** | `1d3b19a937581947400248d8a29e0e3125877bb89d9833cd42dcfe3cb48e7418` |
| **mobile_alamp_v2_fp16.tflite** | `70cbc5c75d5468a3bfd825047f83eda32f4a8da36150f6905013e967c90efa43` |
| **icaa_dat_tf_native_fp16.tflite** | `10ae68a3283bdabf05ecfa8b679e1e89865f3c960687ce5495500455bbd111ff` |

## 4. Troubleshooting
If any checksum does not match:
1. **ICAA**: Check if you have the "tf_native_fp16" version. There are multiple ICAA variants.
2. **NIMA**: Ensure you have the "fixed_preproc" version.
3. **RGNet**: Ensure you have the "pil_resize" version.
4. **Action**: Copy the file from WSL `forWeights2/models/aesthetic/` to Mac to ensure parity.
