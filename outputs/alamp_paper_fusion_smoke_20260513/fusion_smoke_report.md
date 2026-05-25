# MPNet + Layout Graph Fusion Smoke Report

This branch is a paper-oriented approximation, not an official or exact A-LAMP reproduction.

## Files Changed

- `src/models/alamp_paper_fusion.py`
- `src/train/train_alamp_paper_mpnet_graph_fusion.py`
- `checklist.md`
- `context-notes.md`

## Data Files Required

- train patch JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_1024_v4.jsonl`
- val patch JSONL: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_1024_v4.jsonl`
- train graph JSONL: `outputs/alamp_object_graph_subset_20260511/graphs_conf010/train_graphs_1024.jsonl`
- val graph JSONL: `outputs/alamp_object_graph_subset_20260511/graphs_conf010/val_graphs_1024.jsonl`
- MPNet checkpoint: `outputs/alamp_paper_mpnet_selector_compare_20260512/train/v4_1024_gpu_tuned/v4_1024_gpu_tuned/best_val_auc_model.keras`

## Tensor Shapes

```json
{
  "train": {
    "adjacency": [
      8,
      4,
      4
    ],
    "crops": [
      8,
      5,
      224,
      224,
      3
    ],
    "graph_vectors": [
      8,
      100
    ],
    "labels": [
      8,
      1
    ],
    "mpnet_features": [
      8,
      256
    ],
    "node_features": [
      8,
      4,
      15
    ],
    "object_mask": [
      8,
      4
    ],
    "records": 8,
    "skipped_bad_crop": 0,
    "skipped_missing_graph": 0
  },
  "val": {
    "adjacency": [
      4,
      4,
      4
    ],
    "crops": [
      4,
      5,
      224,
      224,
      3
    ],
    "graph_vectors": [
      4,
      100
    ],
    "labels": [
      4,
      1
    ],
    "mpnet_features": [
      4,
      256
    ],
    "node_features": [
      4,
      4,
      15
    ],
    "object_mask": [
      4,
      4
    ],
    "records": 4,
    "skipped_bad_crop": 0,
    "skipped_missing_graph": 0
  }
}
```

## GraphGCN Status

implemented_from_node_features_and_local_edges

## Smoke Command

```bash
PYTHONPATH=. /home/omen_pc1/photo_score_project/.venv_gpu/bin/python src/train/train_alamp_paper_mpnet_graph_fusion.py --smoke --fusion_mode both --max_train_samples 8 --max_val_samples 4 --batch_size 2 --out_dir outputs/alamp_paper_fusion_smoke_20260513
```

## Smoke Result

```json
{
  "graphgcn": {
    "batch_input_shapes": {
      "adjacency": [
        2,
        4,
        4
      ],
      "mpnet_features": [
        2,
        256
      ],
      "node_features": [
        2,
        4,
        15
      ],
      "object_mask": [
        2,
        4
      ]
    },
    "batch_label_shape": [
      2,
      1
    ],
    "fusion_mode": "graphgcn",
    "loss_before_train_step": 0.8771649599075317,
    "prediction_shape": [
      2,
      1
    ],
    "train_on_batch": {
      "accuracy": 1.0,
      "auc": 0.0,
      "loss": 0.43221038579940796
    }
  },
  "graphlite": {
    "batch_input_shapes": {
      "graph_vector": [
        2,
        100
      ],
      "mpnet_features": [
        2,
        256
      ]
    },
    "batch_label_shape": [
      2,
      1
    ],
    "fusion_mode": "graphlite",
    "loss_before_train_step": 2.910539150238037,
    "prediction_shape": [
      2,
      1
    ],
    "train_on_batch": {
      "accuracy": 0.0,
      "auc": 0.0,
      "loss": 3.440403938293457
    }
  }
}
```

## Risks And Unknowns

- Graph cache comes from object detector outputs, not an official A-LAMP detector reproduction.
- GraphLite vectorization is an approximation over fixed graph fields.
- Fusion branch is not yet validated by full 1024 training/evaluation.
- MPNet features are frozen extracted features; this script does not fine-tune MPNet jointly.
