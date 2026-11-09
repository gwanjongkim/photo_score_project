import json
import pandas as pd
from pathlib import Path

configs = [
    "agg_mean",
    "agg_max",
    "agg_lse_r2",
    "agg_lse_r4",
    "agg_lse_r8",
    "dilation_paper_3_6_12_18",
    "input_300",
    "raw_lse_then_sigmoid",
    "graph_blocks_2",
]

results = []
for name in configs:
    train_summary_path = f"outputs/rgnet_paper_v1_ablation_20260509/mid/{name}/train_summary.json"
    eval_summary_path = f"outputs/rgnet_paper_v1_ablation_20260509/eval_mid/{name}/evaluation_summary.json"
    
    try:
        with open(train_summary_path, 'r') as f:
            train_data = json.load(f)
        with open(eval_summary_path, 'r') as f:
            eval_data = json.load(f)
            
        test_metrics = eval_data.get('splits', {}).get('test', {})
        val_loss = train_data.get('best_val_loss')
        
        results.append({
            "Name": name,
            "Test SRCC": test_metrics.get("srcc"),
            "Test PLCC": test_metrics.get("plcc"),
            "Test MAE": test_metrics.get("mae"),
            "Test RMSE": test_metrics.get("rmse"),
            "Val Loss": val_loss,
            "Best Epoch": train_data.get("best_epoch"),
        })
    except Exception as e:
        print(f"Error loading {name}: {e}")

df = pd.DataFrame(results)
df = df.sort_values(by="Test SRCC", ascending=False)
df.to_csv("outputs/rgnet_paper_v1_ablation_20260509/ablation_results.csv", index=False)
with open("outputs/rgnet_paper_v1_ablation_20260509/ablation_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(df.to_markdown(index=False))
