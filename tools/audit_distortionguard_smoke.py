
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    root = Path("/home/omen_pc1/photo_score_project")
    smoke_dir = root / "data/processed/distortionguard_iqa_v1"
    manifest_path = smoke_dir / "synthetic_smoke_manifest.csv"
    pairs_path = smoke_dir / "synthetic_smoke_pairs.csv"
    summary_path = smoke_dir / "synthetic_smoke_summary.json"
    
    out_dir = root / "outputs/distortionguard_iqa_v1_synthetic_smoke_audit_20260523"
    grid_dir = out_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    df_manifest = pd.read_csv(manifest_path)
    df_pairs = pd.read_csv(pairs_path)
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    # 2. Statistics
    print("Gathering statistics...")
    stats = {
        "distortion_types": df_manifest["distortion_type"].unique().tolist(),
        "severities": sorted(df_manifest["severity"].unique().tolist()),
        "counts_by_type": df_manifest["distortion_type"].value_counts().to_dict(),
        "counts_by_severity": df_manifest["severity"].value_counts().to_dict(),
    }
    
    # 3. Pair Logic Check
    # Label 1 means distorted_a is better than distorted_b
    # Pairs are (original, s1), (original, s2)... and (s1, s2), (s1, s3)...
    # original severity is 0
    print("Checking pair logic...")
    invalid_pairs = []
    for idx, row in df_pairs.iterrows():
        sev_a = row["severity_a"]
        sev_b = row["severity_b"]
        label = row["label"]
        
        # Logic: lower severity is better
        # if sev_a < sev_b, label should be 1
        # if sev_a > sev_b, label should be 2 (assuming binary classification better/worse)
        # But looking at the head, label 1 is used for original > s1
        if sev_a < sev_b and label != 1:
            invalid_pairs.append(row.to_dict())
        elif sev_a > sev_b and label != 2:
             # Just in case there are reverse pairs
             invalid_pairs.append(row.to_dict())
             
    # 4. Visual Grid Review
    print("Generating visual grids...")
    source_images = df_manifest["source_image_path"].unique()[:3]
    distortion_types = stats["distortion_types"]
    severities_to_show = [1, 3, 5]
    
    for i, src_path in enumerate(source_images):
        src_name = Path(src_path).name
        
        # Grid dimensions: (len(distortion_types) + 1) rows, (len(severities_to_show) + 1) columns
        # Column 0: Original
        # Columns 1, 2, 3: s1, s3, s5
        
        thumb_size = 300
        padding = 20
        grid_w = (thumb_size + padding) * (len(severities_to_show) + 1)
        grid_h = (thumb_size + padding) * (len(distortion_types))
        
        canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        try:
            original_img = Image.open(src_path).convert('RGB')
            original_img.thumbnail((thumb_size, thumb_size))
        except Exception as e:
            print(f"Error opening {src_path}: {e}")
            continue

        for row_idx, d_type in enumerate(distortion_types):
            # Draw Original at start of every row for comparison
            canvas.paste(original_img, (0, row_idx * (thumb_size + padding)))
            draw.text((10, row_idx * (thumb_size + padding) + 10), f"ORIGINAL\n{d_type}", fill=(255, 0, 0))

            for col_idx, sev in enumerate(severities_to_show):
                subset = df_manifest[(df_manifest["source_image_path"] == src_path) & 
                                     (df_manifest["distortion_type"] == d_type) & 
                                     (df_manifest["severity"] == sev)]
                
                if subset.empty:
                    print(f"Missing image for {d_type} s{sev}")
                    continue
                    
                dist_path = root / subset.iloc[0]["distorted_image_path"]
                try:
                    dist_img = Image.open(dist_path).convert('RGB')
                    dist_img.thumbnail((thumb_size, thumb_size))
                    x = (col_idx + 1) * (thumb_size + padding)
                    y = row_idx * (thumb_size + padding)
                    canvas.paste(dist_img, (x, y))
                    draw.text((x + 10, y + 10), f"s{sev}", fill=(255, 255, 0))
                except Exception as e:
                    print(f"Error opening {dist_path}: {e}")
                    
        canvas.save(grid_dir / f"grid_source_{i}_{src_name}.jpg")
        print(f"Saved grid for {src_name}")

    # 5. Output Results
    audit_results = {
        "stats": stats,
        "invalid_pairs_count": len(invalid_pairs),
        "invalid_pairs_samples": invalid_pairs[:10]
    }
    
    with open(out_dir / "audit_data.json", 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"Audit complete. Results in {out_dir}")

if __name__ == "__main__":
    main()
