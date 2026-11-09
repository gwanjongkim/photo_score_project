import argparse
import subprocess
import sys
from pathlib import Path

configs = [
    "configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/agg_max.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/agg_lse_r2.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/agg_lse_r4.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/agg_lse_r8.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/dilation_paper_3_6_12_18.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/input_300.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/raw_lse_then_sigmoid.yaml",
    "configs/paper_benchmarks/rgnet_v1_ablations/graph_blocks_2.yaml",
]

def main():
    for config in configs:
        name = Path(config).stem
        train_out_dir = f"outputs/rgnet_paper_v1_ablation_20260509/mid/{name}"
        eval_out_dir = f"outputs/rgnet_paper_v1_ablation_20260509/eval_mid/{name}"
        batch_size = "4" if "input_300" in name else "8"

        eval_cmd = [
            sys.executable, "-m", "src.eval.evaluate_rgnet_paper_v1_aadb",
            "--config", config,
            "--model_path", f"{train_out_dir}/final_model.keras",
            "--output_dir", eval_out_dir,
            "--batch_size", batch_size,
            "--max_test_samples", "256"
        ]
        print(f"\n--- Mid Eval: {name} ---")
        subprocess.run(eval_cmd)

if __name__ == "__main__":
    main()
