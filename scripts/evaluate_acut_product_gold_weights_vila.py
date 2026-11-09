import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Tuple, List

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from itertools import product

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Adaptive Patch Selector Logic (v2) ---

def compute_saliency_v2(image_bgr: np.ndarray) -> np.ndarray:
    img_small = cv2.resize(image_bgr, (64, 64))
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(fft) + 1e-6)
    phase = np.angle(fft)
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude
    saliency_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase)))
    saliency_map = saliency_map**2
    saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 0)
    h, w = image_bgr.shape[:2]
    saliency_map = cv2.resize(saliency_map, (w, h))
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    center_bias = np.exp(-((x - cx)**2 / (w / 2.0)**2 + (y - cy)**2 / (h / 2.0)**2))
    saliency_map = saliency_map * (0.8 + 0.2 * center_bias)
    m_min, m_max = saliency_map.min(), saliency_map.max()
    if m_max > m_min:
        saliency_map = (saliency_map - m_min) / (m_max - m_min)
    return saliency_map

def extract_features(image_bgr: np.ndarray, box: np.ndarray) -> dict:
    x1, y1, x2, y2 = map(int, box)
    patch = image_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return {"color_hist": np.zeros(32, dtype=np.float32), "edge_hist": np.zeros(8, dtype=np.float32)}
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    hist_l = cv2.calcHist([patch_lab], [0], None, [16], [0, 256]).flatten()
    hist_a = cv2.calcHist([patch_lab], [1], None, [8], [0, 256]).flatten()
    hist_b = cv2.calcHist([patch_lab], [2], None, [8], [0, 256]).flatten()
    color_hist = np.concatenate([hist_l, hist_a, hist_b]).astype(np.float32)
    color_hist /= (color_hist.sum() + 1e-6)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(sobelx, sobely)
    hist_edge = cv2.calcHist([ang], [0], None, [8], [0, 2 * np.pi]).flatten().astype(np.float32)
    hist_edge /= (hist_edge.sum() + 1e-6)
    return {"color_hist": color_hist, "edge_hist": hist_edge}

def generate_candidate_boxes(width: int, height: int, scales: list, grid_size: int) -> np.ndarray:
    short_edge = min(width, height)
    boxes = []
    for scale in scales:
        size = int(short_edge * scale)
        if size <= 10: continue
        step_x = (width - size) / max(1, grid_size - 1) if width > size else 0
        step_y = (height - size) / max(1, grid_size - 1) if height > size else 0
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = int(i * step_x), int(j * step_y)
                boxes.append([x1, y1, x1 + size, y1 + size])
    return np.array(boxes, dtype=np.float32) if boxes else np.array([[0, 0, width, height]], dtype=np.float32)

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1]), np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = (box[2]-box[0])*(box[3]-box[1]) + (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-6)

def select_patches_v2(image_bgr: np.ndarray, patch_count: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    scales = [0.25, 0.35, 0.5]
    grid_size = 8
    boxes = generate_candidate_boxes(w, h, scales, grid_size)
    saliency_map = compute_saliency_v2(image_bgr)
    box_scores = np.array([saliency_map[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean() if saliency_map[int(b[1]):int(b[3]), int(b[0]):int(b[2])].size > 0 else 0.0 for b in boxes], dtype=np.float32)
    box_feats = [extract_features(image_bgr, b) for b in boxes]
    box_centers = np.array([[(b[0]+b[2])/2.0/w, (b[1]+b[3])/2.0/h] for b in boxes], dtype=np.float32)
    
    selected_indices = []
    w_saliency, w_pattern, w_spatial, iou_threshold = 1.0, 0.5, 0.3, 0.5
    for _ in range(patch_count):
        if not selected_indices:
            best_idx = np.argmax(box_scores)
        else:
            scores = w_saliency * box_scores.copy()
            for idx in selected_indices:
                ious = compute_iou(boxes[idx], boxes)
                scores[ious > iou_threshold] = -1.0
                curr_feat = box_feats[idx]
                for i, other_feat in enumerate(box_feats):
                    if scores[i] <= 0: continue
                    color_dist = cv2.compareHist(curr_feat["color_hist"], other_feat["color_hist"], cv2.HISTCMP_BHATTACHARYYA)
                    edge_dist = cv2.compareHist(curr_feat["edge_hist"], other_feat["edge_hist"], cv2.HISTCMP_BHATTACHARYYA)
                    scores[i] += w_pattern * (color_dist + edge_dist)
                dist = np.sqrt(np.sum((box_centers - box_centers[idx])**2, axis=1))
                scores += w_spatial * dist
            scores[selected_indices] = -1.0
            best_idx = np.argmax(scores)
        selected_indices.append(best_idx)
    return boxes[selected_indices], box_scores[selected_indices]

# --- Model Wrapper ---

class AestheticEnsembleEvaluator:
    def __init__(self, nima_path, rgnet_path, alamp_path):
        self.nima = tf.lite.Interpreter(model_path=nima_path)
        self.nima.allocate_tensors()
        self.rgnet = tf.lite.Interpreter(model_path=rgnet_path)
        self.rgnet.allocate_tensors()
        self.alamp = tf.lite.Interpreter(model_path=alamp_path)
        self.alamp.allocate_tensors()
        
    def predict_nima(self, img_pil: Image.Image) -> float:
        img = img_pil.resize((224, 224), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        input_details = self.nima.get_input_details()
        output_details = self.nima.get_output_details()
        self.nima.set_tensor(input_details[0]['index'], arr)
        self.nima.invoke()
        probs = self.nima.get_tensor(output_details[0]['index'])[0]
        raw10 = sum(probs[i] * (i + 1) for i in range(10))
        return (raw10 - 1.0) / 9.0

    def predict_rgnet(self, img_pil: Image.Image) -> float:
        img = img_pil.resize((256, 256), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        input_details = self.rgnet.get_input_details()
        output_details = self.rgnet.get_output_details()
        self.rgnet.set_tensor(input_details[0]['index'], arr)
        self.rgnet.invoke()
        return float(self.rgnet.get_tensor(output_details[0]['index'])[0][0])

    def predict_alamp(self, img_pil: Image.Image) -> float:
        img_384 = img_pil.resize((384, 384), Image.BILINEAR)
        arr_full = np.array(img_384).astype(np.float32)
        arr_full = np.expand_dims(arr_full, axis=0)
        
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        selected_boxes, _ = select_patches_v2(img_bgr)
        
        patches = []
        for box in selected_boxes:
            x1, y1, x2, y2 = map(int, box)
            patch = img_bgr[y1:y2, x1:x2]
            patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patches.append(patch_rgb.astype(np.float32))
        arr_patches = np.expand_dims(np.stack(patches), axis=0)
        
        input_details = self.alamp.get_input_details()
        output_details = self.alamp.get_output_details()
        for detail in input_details:
            if detail['shape'][1] == 384:
                self.alamp.set_tensor(detail['index'], arr_full)
            elif detail['shape'][1] == 5:
                self.alamp.set_tensor(detail['index'], arr_patches)
        self.alamp.invoke()
        return float(self.alamp.get_tensor(output_details[0]['index'])[0][0])

def normalize_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if s.max() == s.min(): return s * 0.0 + 1.0
    if higher_is_better:
        return (s - s.min()) / (s.max() - s.min())
    else:
        return (s.max() - s) / (s.max() - s.min())

def get_spread_quality(spread: float) -> float:
    if 0.25 <= spread <= 0.55: return 1.0
    if spread < 0.10 or spread > 0.75: return 0.0
    if spread < 0.25: return (spread - 0.10) / (0.25 - 0.10)
    return 1.0 - (spread - 0.55) / (0.75 - 0.55)

def simple_table(df, cols):
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        row_vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row_vals.append(f"{v:.4f}")
            else:
                row_vals.append(str(v))
        rows.append("| " + " | ".join(row_vals) + " |")
    return "\n".join([header, sep] + rows)

def evaluate_weights(df, df_valid_for_pw, w_nima, w_rgnet, w_alamp):
    scores = df['nima_norm'] * w_nima + df['rgnet_norm'] * w_rgnet + df['alamp_norm'] * w_alamp
    srcc, _ = spearmanr(df['label_ordinal_1_3'], scores)
    plcc, _ = pearsonr(df['label_norm_0_1'], scores)
    mae = np.mean(np.abs(df['label_norm_0_1'] - scores))
    
    means = df.assign(score=scores).groupby('label_name')['score'].mean()
    bad_m = means.get('Bad', np.nan)
    soso_m = means.get('Soso', np.nan)
    good_m = means.get('Good', np.nan)
    
    # Pairwise accuracy on valid set
    pairs = 0
    correct = 0
    for i, r1 in df_valid_for_pw.iterrows():
        for j, r2 in df_valid_for_pw.iterrows():
            if r1['label_norm_0_1'] > r2['label_norm_0_1']:
                pairs += 1
                s1 = r1['nima_norm']*w_nima + r1['rgnet_norm']*w_rgnet + r1['alamp_norm']*w_alamp
                s2 = r2['nima_norm']*w_nima + r2['rgnet_norm']*w_rgnet + r2['alamp_norm']*w_alamp
                if s1 > s2: correct += 1
    pw_acc = correct / max(pairs, 1)
    
    p10, p90 = np.percentile(scores, 10), np.percentile(scores, 90)
    
    return {
        "srcc": srcc, "plcc": plcc, "mae": mae,
        "bad_mean": bad_m, "soso_mean": soso_m, "good_mean": good_m,
        "good_minus_bad": good_m - bad_m,
        "sep_quality": (good_m - soso_m) + (soso_m - bad_m),
        "pw_acc": pw_acc,
        "std": np.std(scores), "p90_minus_p10": p90 - p10
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", default="test_vila/")
    parser.add_argument("--out_dir", default="outputs/acut_product_gold_weight_eval_test_vila_20260603/")
    parser.add_argument("--prev_preds", default="outputs/acut_product_gold_weight_eval_20260603/product_gold_predictions.csv")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Step 1: Inventory
    images = []
    night_keywords = ["night", "sky", "city", "skyline", "star", "moon", "dark", "야경", "밤", "하늘"]
    for label in ["Bad", "Soso", "Good"]:
        d = os.path.join(args.gold_dir, label)
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')):
                tag = "unknown"
                fname_lower = f.lower()
                if any(k in fname_lower for k in night_keywords):
                    tag = "night_or_sky"
                
                images.append({
                    "image_path": os.path.join(d, f),
                    "label_name": label,
                    "label_ordinal_1_3": {"Bad": 1, "Soso": 2, "Good": 3}[label],
                    "label_norm_0_1": {"Bad": 0.0, "Soso": 0.5, "Good": 1.0}[label],
                    "filename": f,
                    "suspected_domain_tag": tag
                })
    
    df_manifest = pd.DataFrame(images)
    df_manifest.to_csv(os.path.join(args.out_dir, "test_vila_manifest.csv"), index=False)
    logging.info(f"Inventory complete: {len(images)} images found.")
    
    # Step 2: Predictions
    evaluator = AestheticEnsembleEvaluator(
        "forWeights2/models/aesthetic/nima_mobile_fixed_preproc_fp16.tflite",
        "forWeights2/models/aesthetic/rgnet_pil_resize_aadb_fp16.tflite",
        "forWeights2/models/aesthetic/mobile_alamp_v2_fp16.tflite"
    )
    
    preds = []
    failures = []
    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest)):
        try:
            with Image.open(row['image_path']) as img:
                img = img.convert("RGB")
                img = ImageOps.exif_transpose(img)
                nima = evaluator.predict_nima(img)
                rgnet = evaluator.predict_rgnet(img)
                alamp = evaluator.predict_alamp(img)
                preds.append({**row, "nima_norm": nima, "rgnet_norm": rgnet, "alamp_norm": alamp, "failure_reason": None})
        except Exception as e:
            preds.append({**row, "nima_norm": np.nan, "rgnet_norm": np.nan, "alamp_norm": np.nan, "failure_reason": str(e)})
            failures.append({**row, "error": str(e)})
            
    df_preds = pd.DataFrame(preds)
    df_preds.to_csv(os.path.join(args.out_dir, "test_vila_predictions.csv"), index=False)
    if failures: pd.DataFrame(failures).to_csv(os.path.join(args.out_dir, "failures.csv"), index=False)
    df_valid = df_preds.dropna(subset=["nima_norm", "rgnet_norm", "alamp_norm"]).copy()
    
    # Step 3: Fixed Candidates
    candidates = [
        ("current_v3_alamp_heavy", 0.25, 0.30, 0.45),
        ("product_gold_winner", 0.40, 0.45, 0.15),
        ("multi_objective_candidate", 0.47, 0.23, 0.30),
        ("old_no_icaa_renorm", 0.125, 0.50, 0.375),
        ("new_uncalibrated", 0.37, 0.41, 0.22),
        ("rgnet_heavy", 0.20, 0.55, 0.25),
        ("alamp_very_heavy", 0.20, 0.20, 0.60),
        ("nima_heavy", 0.50, 0.30, 0.20),
    ]
    
    fixed_res = []
    for name, wn, wr, wa in candidates:
        metrics = evaluate_weights(df_valid, df_valid, wn, wr, wa)
        fixed_res.append({"candidate_name": name, "w_nima": wn, "w_rgnet": wr, "w_alamp": wa, **metrics})
    pd.DataFrame(fixed_res).to_csv(os.path.join(args.out_dir, "fixed_candidate_metrics.csv"), index=False)
    
    # Step 4: Weight Search on test_vila
    search_res = []
    for wn in np.arange(0, 1.0001, 0.05):
        for wr in np.arange(0, 1.0001 - wn, 0.05):
            wa = 1.0 - wn - wr
            m = evaluate_weights(df_valid, df_valid, wn, wr, wa)
            search_res.append({"w_nima": wn, "w_rgnet": wr, "w_alamp": wa, **m})
    df_search = pd.DataFrame(search_res)
    
    def calc_obj(df):
        df['srcc_norm'] = normalize_series(df['srcc'])
        df['pw_acc_norm'] = normalize_series(df['pw_acc'])
        df['sep_norm'] = normalize_series(df['sep_quality'])
        df['plcc_norm'] = normalize_series(df['plcc'])
        df['mae_inv_norm'] = normalize_series(df['mae'], False)
        df['spread_qual'] = df['p90_minus_p10'].apply(get_spread_quality)
        df['spread_norm'] = normalize_series(df['spread_qual'])
        return (0.35 * df['srcc_norm'] + 0.25 * df['pw_acc_norm'] + 0.15 * df['sep_norm'] +
                0.10 * df['plcc_norm'] + 0.10 * df['mae_inv_norm'] + 0.05 * df['spread_norm'])

    df_search['product_objective'] = calc_obj(df_search)
    df_search.to_csv(os.path.join(args.out_dir, "weight_search_results.csv"), index=False)
    top_vila = df_search.nlargest(20, 'product_objective')
    top_vila.to_csv(os.path.join(args.out_dir, "top_weight_candidates.csv"), index=False)
    
    # Step 5: Pooling
    pooled_winner_info = None
    if os.path.exists(args.prev_preds):
        df_prev = pd.read_csv(args.prev_preds)
        # Combine
        common_cols = ['label_name', 'label_ordinal_1_3', 'label_norm_0_1', 'nima_norm', 'rgnet_norm', 'alamp_norm']
        df_pooled = pd.concat([df_valid[common_cols], df_prev[common_cols]]).dropna()
        
        pooled_search = []
        for wn in np.arange(0, 1.0001, 0.05):
            for wr in np.arange(0, 1.0001 - wn, 0.05):
                wa = 1.0 - wn - wr
                m = evaluate_weights(df_pooled, df_pooled, wn, wr, wa)
                pooled_search.append({"w_nima": wn, "w_rgnet": wr, "w_alamp": wa, **m})
        df_pooled_search = pd.DataFrame(pooled_search)
        df_pooled_search['pooled_objective'] = calc_obj(df_pooled_search)
        pooled_winner = df_pooled_search.nlargest(1, 'pooled_objective').iloc[0]
        pooled_winner_info = pooled_winner
        df_pooled_search.to_csv(os.path.join(args.out_dir, "pooled_weight_search_results.csv"), index=False)

    # Step 6: Caveat Analysis
    df_night = df_valid[df_valid['suspected_domain_tag'] == 'night_or_sky']
    if not df_night.empty:
        night_stats = df_night.groupby('label_name').agg({
            'nima_norm': ['mean', 'count'],
            'rgnet_norm': 'mean',
            'alamp_norm': 'mean'
        })
        night_stats.to_csv(os.path.join(args.out_dir, "domain_caveat_analysis.csv"))
        
    class_means = df_valid.groupby('label_name').agg({'nima_norm':'mean', 'rgnet_norm':'mean', 'alamp_norm':'mean'})
    class_means.to_csv(os.path.join(args.out_dir, "class_mean_scores.csv"))

    # Step 7: Report
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("# Product Gold Dataset Evaluation: test_vila\n\n")
        f.write("## 1. Executive Summary\n")
        f.write(f"The aesthetic ensemble weights were evaluated on the 'test_vila' dataset ({len(df_valid)} images). ")
        winner_vila = top_vila.iloc[0]
        f.write(f"Standalone winner for test_vila: **NIMA {winner_vila['w_nima']:.2f} / RGNet {winner_vila['w_rgnet']:.2f} / A-LAMP {winner_vila['w_alamp']:.2f}**.\n\n")
        
        f.write("## 2. Dataset Counts\n")
        counts = df_preds['label_name'].value_counts()
        f.write(f"- Bad: {counts.get('Bad', 0)}\n")
        f.write(f"- Soso: {counts.get('Soso', 0)}\n")
        f.write(f"- Good: {counts.get('Good', 0)}\n")
        f.write(f"- Failures: {len(failures)}\n\n")
        
        f.write("## 3. Fixed Candidate Comparison (test_vila)\n")
        f.write(simple_table(pd.DataFrame(fixed_res), ['candidate_name', 'w_nima', 'w_rgnet', 'w_alamp', 'srcc', 'pw_acc', 'good_minus_bad']) + "\n\n")
        
        f.write("## 4. Night/Sky Caveat Analysis\n")
        f.write(f"- Count of night_or_sky: {len(df_night)}\n")
        if not df_night.empty:
            f.write("Night/Sky images (especially Good ones) often show different score distributions. ")
            f.write("If Good night images are under-scored, it may be due to low saliency or lack of ordinary 'beauty' patterns in AVA/AADB training.\n\n")
            f.write(simple_table(df_night[['filename', 'label_name', 'nima_norm', 'rgnet_norm', 'alamp_norm']], ['filename', 'label_name', 'nima_norm', 'rgnet_norm', 'alamp_norm']) + "\n\n")

        if pooled_winner_info is not None:
            f.write("## 5. Pooled Results (Test_weight + test_vila)\n")
            f.write(f"Pooled winner: **NIMA {pooled_winner_info['w_nima']:.2f} / RGNet {pooled_winner_info['w_rgnet']:.2f} / A-LAMP {pooled_winner_info['w_alamp']:.2f}**.\n\n")

        f.write("## 6. Final Recommendation\n")
        prev_winner = [0.40, 0.45, 0.15]
        v3 = [0.25, 0.30, 0.45]
        
        f.write(f"Previous winner (0.40/0.45/0.15) SRCC on test_vila: {fixed_res[1]['srcc']:.4f}\n")
        f.write(f"Current V3 (0.25/0.30/0.45) SRCC on test_vila: {fixed_res[0]['srcc']:.4f}\n\n")
        
        if pooled_winner_info is not None:
            f.write(f"**RECOMMENDED**: Use pooled winner **NIMA {pooled_winner_info['w_nima']:.2f} / RGNet {pooled_winner_info['w_rgnet']:.2f} / A-LAMP {pooled_winner_info['w_alamp']:.2f}** as it generalizes across both gold sets.\n")
        else:
            f.write("**RECOMMENDED**: Stick to **0.40/0.45/0.15** until more general portrait data is collected.\n")

    print(f"Report saved to: {args.out_dir}report.md")

if __name__ == "__main__":
    main()
