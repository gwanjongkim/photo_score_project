# ICAA17K Export-Safe TorchScript Export and Parity
import os
import sys
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import json
from datetime import datetime
import argparse
import warnings

def get_model_kwargs():
    return dict(
        img_size=224,
        patch_size=4,
        num_classes=1,
        expansion=4,
        dim_stem=128,
        dims=[128, 256, 512, 1024],
        depths=[2, 2, 18, 2],
        stage_spec=[["L","S"], ["L","S"], ["L","D"] * 9, ["L","D"]],
        heads=[4, 8, 16, 32],
        window_sizes=[7, 7, 7, 7],
        groups=[-1, -1, 4, 8],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.5
    )

class DATWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    
    checkpoint_path = 'weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth'
    test_csv = 'external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv'
    data_root = 'data/raw/icaa17k/'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/icaa_export_safe_torchscript_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    export_safe_parent_path = os.path.abspath('experiments/icaa_export_safe')
    sys.path.insert(0, export_safe_parent_path)
    from export_safe_models.dat import DAT
    
    model = DAT(**get_model_kwargs())
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()
    
    wrapper = DATWrapper(model)
    wrapper.eval()
    
    print("Tracing model...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Catch TracerWarnings
    with warnings.catch_warnings(record=True) as w:
        traced_model = torch.jit.trace(wrapper, dummy_input)
        
        traced_path = os.path.join(output_dir, 'icaa_dat_export_safe_traced.pt')
        traced_model.save(traced_path)
        print(f"Traced model saved to {traced_path}")
        
        tracer_warnings = []
        for warning in w:
            if "TracerWarning" in str(warning.category):
                tracer_warnings.append(str(warning.message))
                print(f"TracerWarning: {warning.message}")

    # Parity check
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df = pd.read_csv(test_csv)
    if args.limit > 0:
        df = df.head(args.limit)
    
    results = []
    max_diff = 0.0
    sum_diff = 0.0
    count = 0
    
    print(f"Running parity on {len(df)} images...")
    
    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            batch_df = df.iloc[i : i + args.batch_size]
            batch_tensors = []
            for _, row in batch_df.iterrows():
                img_path = os.path.join(data_root, row['ID'])
                img = Image.open(img_path).convert('RGB')
                batch_tensors.append(transform(img))
            
            x = torch.stack(batch_tensors).to(device)
            
            out_py = wrapper(x)
            out_ts = traced_model(x)
            
            diff = torch.abs(out_py - out_ts)
            batch_max = diff.max().item()
            batch_mean = diff.mean().item()
            
            max_diff = max(max_diff, batch_max)
            sum_diff += diff.sum().item()
            count += diff.numel()
            
            for j, (_, row) in enumerate(batch_df.iterrows()):
                results.append({
                    'ID': row['ID'],
                    'py_mos': out_py[j, 0].item(),
                    'ts_mos': out_ts[j, 0].item(),
                    'diff': diff[j, 0].item()
                })
    
    mean_diff = sum_diff / count
    
    report = {
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'pass_threshold_1e-5': max_diff <= 1e-5,
        'pass_threshold_1e-4': max_diff <= 1e-4,
        'num_samples': len(df),
        'tracer_warnings': tracer_warnings
    }
    
    with open(os.path.join(output_dir, 'export_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'parity_predictions.csv'), index=False)
    
    print(f"PyTorch vs TorchScript Parity Results:")
    print(f"  Max Abs Diff: {max_diff:.10f}")
    print(f"  Mean Abs Diff: {mean_diff:.10f}")
    if report['pass_threshold_1e-5']:
        print("  Status: PASS (1e-5)")
    elif report['pass_threshold_1e-4']:
        print("  Status: PASS (1e-4)")
    else:
        print("  Status: FAIL")

if __name__ == '__main__':
    main()
