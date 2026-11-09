# Mining TechIQA-Guard Hard False Positive Candidates
import pandas as pd
import numpy as np
from pathlib import Path
import json

def main():
    root_dir = Path("/home/omen_pc1/photo_score_project")
    out_dir = root_dir / "data/processed/techiqa_guard/fp_mining_20260522"
    out_dir.mkdir(parents=True, exist_ok=True)

    stage4_dir = root_dir / "outputs/eval_techiqa_guard_v1_stage4_fpdata_lam005_20260520"
    stage5_dir = root_dir / "outputs/eval_techiqa_guard_v1_stage5_rank002_fpguard005_v2_20260520"
    
    datasets = ["flive", "koniq", "spaq", "smoke", "hard_fp"]
    
    all_rows = []

    for ds in datasets:
        # Load existing technical combo
        p_existing = stage4_dir / f"predictions_existing_avg_technical_{ds}.csv"
        if not p_existing.exists():
            continue
            
        df_existing = pd.read_csv(p_existing)
        # Columns: image_path, y_true_k, y_pred_k, y_true_f, y_pred_f, y_pred, y_true
        df_existing = df_existing.rename(columns={
            "y_pred_k": "koniq_mobile_score",
            "y_pred_f": "flive_image_mobile_score",
            "y_pred": "existing_combined_score"
        })
        df_existing = df_existing[["image_path", "koniq_mobile_score", "flive_image_mobile_score", "existing_combined_score"]]
        
        def load_pred(path, col_name):
            if path.exists():
                tmp = pd.read_csv(path)
                return tmp.rename(columns={"y_pred": col_name})[["image_path", col_name]]
            return pd.DataFrame(columns=["image_path", col_name])

        # Load mixed112
        df_mixed = load_pred(stage4_dir / f"predictions_topiq_lite_mixed112_frozen_fp16_{ds}.csv", "mixed112_score")
        
        # Load ranking_lam01
        df_ranking = load_pred(stage4_dir / f"predictions_topiq_lite_ranking_lam01_gap05_fp16_{ds}.csv", "ranking_lam01_score")
        
        # Load TechIQA Stage 4
        df_s4 = load_pred(stage4_dir / f"predictions_techiqa_guard_v1_s1_frozen_{ds}.csv", "techiqa_stage4_score")
        
        # Load TechIQA Stage 5
        df_s5 = load_pred(stage5_dir / f"predictions_techiqa_guard_v1_s1_frozen_{ds}.csv", "techiqa_stage5_score")

        # Merge
        merged = df_existing.merge(df_mixed, on="image_path", how="left")
        merged = merged.merge(df_ranking, on="image_path", how="left")
        merged = merged.merge(df_s4, on="image_path", how="left")
        merged = merged.merge(df_s5, on="image_path", how="left")
        
        merged["dataset"] = ds
        all_rows.append(merged)

    master_df = pd.concat(all_rows, ignore_index=True)
    
    # Calculate deltas
    master_df["delta_mixed112_existing"] = master_df["mixed112_score"] - master_df["existing_combined_score"]
    master_df["delta_ranking_existing"] = master_df["ranking_lam01_score"] - master_df["existing_combined_score"]
    master_df["delta_stage5_stage4"] = master_df["techiqa_stage5_score"] - master_df["techiqa_stage4_score"]
    
    # Candidate rules
    master_df["hard_fp_candidate"] = (master_df["delta_mixed112_existing"] >= 5.0) | (master_df["delta_ranking_existing"] >= 5.0)
    master_df["strong_hard_fp_candidate"] = (master_df["delta_mixed112_existing"] >= 8.0) | (master_df["delta_ranking_existing"] >= 8.0)
    
    # Known manual files
    manual_files = ["20230201_181300.jpg", "1675342165226-13.jpg", "1675342165226-3.jpg"]
    master_df["filename"] = master_df["image_path"].apply(lambda x: Path(x).name)
    master_df["manual_false_positive"] = master_df["filename"].isin(manual_files)
    
    # confirmed_false_positive
    # Also include anything already in hard_fp dataset as confirmed
    master_df["confirmed_false_positive"] = (master_df["manual_false_positive"]) | (master_df["dataset"] == "hard_fp")
    
    # Missing image check
    master_df["missing_image"] = master_df["image_path"].apply(lambda x: not Path(x).is_file())
    
    # Source note
    master_df["source_note"] = master_df["dataset"].apply(lambda x: f"mining_from_{x}")
    master_df.loc[master_df["manual_false_positive"], "source_note"] = "manual_user_flagged"
    
    # Sort by delta descending
    master_df = master_df.sort_values(by="delta_mixed112_existing", ascending=False)
    
    # Filter for reportable rows (at least one delta >= 5 or confirmed)
    report_df = master_df[(master_df["hard_fp_candidate"]) | (master_df["confirmed_false_positive"])].copy()
    
    # Create required files
    report_df.to_csv(out_dir / "hard_false_positive_candidates.csv", index=False)
    
    confirmed_df = report_df[report_df["confirmed_false_positive"]].copy()
    confirmed_df.to_csv(out_dir / "hard_false_positive_confirmed.csv", index=False)
    
    # Visual review queue: top 50 deltas that are not confirmed yet
    queue_df = report_df[~report_df["confirmed_false_positive"]].head(50).copy()
    queue_df.to_csv(out_dir / "visual_review_queue.csv", index=False)
    
    # Summary
    summary = {
        "total_mined_rows": len(master_df),
        "hard_fp_candidates": int(master_df["hard_fp_candidate"].sum()),
        "strong_hard_fp_candidates": int(master_df["strong_hard_fp_candidate"].sum()),
        "confirmed_false_positives": int(master_df["confirmed_false_positive"].sum()),
        "missing_images_in_candidates": int(report_df["missing_image"].sum()),
        "dataset_counts": master_df["dataset"].value_counts().to_dict(),
        "candidate_counts_by_dataset": report_df["dataset"].value_counts().to_dict()
    }
    
    with open(out_dir / "fp_mining_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Mining complete. Summary: {summary}")

if __name__ == "__main__":
    main()
