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

def run_stage(stage, extra_args):
    print(f"=== Running Stage: {stage} ===")
    for config in configs:
        name = Path(config).stem
        out_dir = f"outputs/rgnet_paper_v1_ablation_20260509/{stage}/{name}"
        
        cmd = [
            sys.executable, "-m", "src.train.train_rgnet_paper_v1_aadb",
            "--config", config,
            "--out_dir", out_dir
        ] + extra_args
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAIL: {name} failed at {stage}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "mid", "eval_mid"], required=True)
    args = parser.parse_args()
    
    if args.stage == "smoke":
        extra_args = [
            "--max_train_samples", "32",
            "--max_val_samples", "16",
            "--epochs", "1",
            "--batch_size", "2",
            "--backbone_weights", "none"
        ]
        run_stage("smoke", extra_args)
    elif args.stage == "mid":
        for config in configs:
            name = Path(config).stem
            train_out_dir = f"outputs/rgnet_paper_v1_ablation_20260509/mid/{name}"
            eval_out_dir = f"outputs/rgnet_paper_v1_ablation_20260509/eval_mid/{name}"
            batch_size = "4" if "input_300" in name else "8"
            
            # Train
            train_cmd = [
                sys.executable, "-m", "src.train.train_rgnet_paper_v1_aadb",
                "--config", config,
                "--out_dir", train_out_dir,
                "--max_train_samples", "1024",
                "--max_val_samples", "256",
                "--epochs", "5",
                "--batch_size", batch_size,
                "--backbone_weights", "imagenet"
            ]
            print(f"\n--- Mid Train: {name} ---")
            res = subprocess.run(train_cmd)
            if res.returncode != 0:
                print(f"FAIL: {name} failed mid train")
                continue
                
            # Eval
            eval_cmd = [
                sys.executable, "-m", "src.eval.evaluate_rgnet_paper_v1_aadb",
                "--config", config,
                "--model_path", f"{train_out_dir}/best.weights.h5",
                "--output_dir", eval_out_dir,
                "--batch_size", batch_size,
                "--max_test_samples", "256"
            ]
            print(f"\n--- Mid Eval: {name} ---")
            subprocess.run(eval_cmd)

if __name__ == "__main__":
    main()
