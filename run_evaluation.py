import os
import glob
import json
import time
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

# 1. Setup paths
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/test_vila_aesthetic_direct_compare_{TIMESTAMP}"
THUMBS_DIR = os.path.join(OUTPUT_DIR, "thumbs")
os.makedirs(THUMBS_DIR, exist_ok=True)

PREV_CSV = "outputs/test_vila_aesthetic_direct_compare_20260504_135316/test_vila_aesthetic_comparison.csv"
MODEL_PATH = "exports/tflite/stage5_student_full_aadb_aesthetic_mobile_fp32.tflite"

# 2. Load previous data
prev_df = pd.read_csv(PREV_CSV)
prev_data = prev_df.set_index('image_name').to_dict('index')

# 3. Setup TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. Run inference
results = []
image_paths = sorted(glob.glob("test_vila/*.jpg") + glob.glob("test_vila/*.JPG") + glob.glob("test_vila/*.png"))

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    
    # Preprocessing
    try:
        img_raw = tf.io.read_file(img_path)
        img_tensor = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, [224, 224]) # default is bilinear
        img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
        img_tensor = tf.expand_dims(img_tensor, 0)
    except Exception as e:
        print(f"TF decode failed for {img_name}: {e}. Falling back to Pillow.")
        with Image.open(img_path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = im.resize((224, 224), Image.Resampling.BILINEAR)
            img_tensor = np.array(im).astype(np.float32) / 255.0
            img_tensor = np.expand_dims(img_tensor, 0)
            img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    student_score = float(np.clip(output_data[0][0], 0.0, 1.0))
    
    # Get previous data
    row = prev_data.get(img_name, {})
    
    results.append({
        'image_name': img_name,
        'image_path': img_path,
        'old_weighted_aesthetic_score': row.get('old_weighted_aesthetic_score', np.nan),
        'nima_unit_score': row.get('nima_unit_score', np.nan),
        'rgnet_score': row.get('rgnet_score', np.nan),
        'alamp_score': row.get('alamp_score', np.nan),
        'stage5_student_score': student_score,
        'stage5_reference_score': row.get('stage5_reference_score', np.nan),
        'error': ''
    })
    
    # Thumbnails
    safe_filename = img_name.replace(" ", "_")
    thumb_path = os.path.join(THUMBS_DIR, safe_filename)
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((256, 256))
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(thumb_path, "JPEG")

df = pd.DataFrame(results)

# Calculate ranks
df['old_weighted_rank'] = df['old_weighted_aesthetic_score'].rank(method='min', ascending=False)
df['stage5_student_rank'] = df['stage5_student_score'].rank(method='min', ascending=False)
df['stage5_reference_rank'] = df['stage5_reference_score'].rank(method='min', ascending=False)

def get_disagreement(delta):
    if pd.isna(delta): return ""
    if delta >= 10: return "high"
    if delta >= 5: return "medium"
    return "low"

df['rank_delta_old_vs_student'] = (df['old_weighted_rank'] - df['stage5_student_rank']).abs()
df['rank_delta_student_vs_reference'] = (df['stage5_student_rank'] - df['stage5_reference_rank']).abs()
df['rank_delta_old_vs_reference'] = (df['old_weighted_rank'] - df['stage5_reference_rank']).abs()

df['disagreement_type_old_vs_student'] = df['rank_delta_old_vs_student'].apply(get_disagreement)
df['disagreement_type_student_vs_reference'] = df['rank_delta_student_vs_reference'].apply(get_disagreement)

# Reorder columns as requested
cols = [
    'image_name', 'image_path', 'old_weighted_aesthetic_score', 'old_weighted_rank',
    'nima_unit_score', 'rgnet_score', 'alamp_score', 
    'stage5_student_score', 'stage5_student_rank', 
    'stage5_reference_score', 'stage5_reference_rank',
    'rank_delta_old_vs_student', 'rank_delta_student_vs_reference', 'rank_delta_old_vs_reference',
    'disagreement_type_old_vs_student', 'disagreement_type_student_vs_reference', 'error'
]
df = df[cols]
df.to_csv(os.path.join(OUTPUT_DIR, "test_vila_aesthetic_comparison.csv"), index=False)

# Summary JSON
spearman_old_vs_student = df[['old_weighted_aesthetic_score', 'stage5_student_score']].corr(method='spearman').iloc[0,1]
spearman_student_vs_reference = df[['stage5_student_score', 'stage5_reference_score']].corr(method='spearman').iloc[0,1]
spearman_old_vs_reference = df[['old_weighted_aesthetic_score', 'stage5_reference_score']].corr(method='spearman').iloc[0,1]

def get_top5(col):
    return df.nsmallest(5, col)['image_name'].tolist()

top5_old = get_top5('old_weighted_rank')
top5_student = get_top5('stage5_student_rank')
top5_ref = get_top5('stage5_reference_rank')

top1_old = top5_old[0] if top5_old else ""
top1_student = top5_student[0] if top5_student else ""
top1_stage5_reference = top5_ref[0] if top5_ref else ""

summary = {
    "image_count": int(len(df)),
    "valid_image_count": int(df['stage5_student_score'].notna().sum()),
    "old_model_runnable": True,
    "stage5_student_runnable": True,
    "stage5_reference_runnable": True,
    "spearman_old_vs_student": float(spearman_old_vs_student),
    "spearman_student_vs_reference": float(spearman_student_vs_reference),
    "spearman_old_vs_reference": float(spearman_old_vs_reference),
    "top1_old": top1_old,
    "top1_student": top1_student,
    "top1_stage5_reference": top1_stage5_reference,
    "top5_old": top5_old,
    "top5_student": top5_student,
    "top5_stage5_reference": top5_ref,
    "top5_overlap_old_vs_student": int(len(set(top5_old) & set(top5_student))),
    "top5_overlap_student_vs_reference": int(len(set(top5_student) & set(top5_ref))),
    "top5_overlap_old_vs_reference": int(len(set(top5_old) & set(top5_ref))),
    "high_disagreement_count_old_vs_student": int((df['disagreement_type_old_vs_student'] == 'high').sum()),
    "high_disagreement_count_student_vs_reference": int((df['disagreement_type_student_vs_reference'] == 'high').sum()),
    "high_disagreement_count_old_vs_reference": int((df['rank_delta_old_vs_reference'] >= 10).sum())
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Generate HTML
html = f"""
<html>
<head>
<style>
body {{ font-family: sans-serif; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
img {{ max-width: 200px; }}
.high {{ background-color: #ffcccc; }}
.medium {{ background-color: #ffffcc; }}
.low {{ background-color: #ccffcc; }}
.guide {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Aesthetic Direct Comparison</h1>

<div class="guide">
<h3>Human Review Guide</h3>
<p>For high-disagreement images, ask:</p>
<ul>
<li>Which rank feels more correct by human eye?</li>
<li>Is the difference caused by composition, subject clarity, lighting, blur, expression, background, or taste?</li>
<li>Does stage5 student approximate stage5_reference well enough for mobile use?</li>
</ul>
</div>

<h2>Top 5 by old ensemble</h2>
<ol>{''.join([f'<li>{x}</li>' for x in top5_old])}</ol>

<h2>Top 5 by stage5 student</h2>
<ol>{''.join([f'<li>{x}</li>' for x in top5_student])}</ol>

<h2>Top 5 by stage5_reference</h2>
<ol>{''.join([f'<li>{x}</li>' for x in top5_ref])}</ol>

<h2>Highest Disagreement: Old vs Student</h2>
<ol>{''.join([f'<li>{x}</li>' for x in df.nlargest(5, "rank_delta_old_vs_student")["image_name"]])}</ol>

<h2>Highest Disagreement: Student vs Reference</h2>
<ol>{''.join([f'<li>{x}</li>' for x in df.nlargest(5, "rank_delta_student_vs_reference")["image_name"]])}</ol>

<h2>Highest Disagreement: Old vs Reference</h2>
<ol>{''.join([f'<li>{x}</li>' for x in df.nlargest(5, "rank_delta_old_vs_reference")["image_name"]])}</ol>

<table>
<tr>
  <th>Thumbnail</th>
  <th>Image Name</th>
  <th>Old Score/Rank</th>
  <th>NIMA/RGNet/A-Lamp</th>
  <th>Student Score/Rank</th>
  <th>Reference Score/Rank</th>
  <th>Rank Deltas (O-S, S-R)</th>
  <th>Disagreements</th>
  <th>Human Note</th>
</tr>
"""

for _, row in df.iterrows():
    safe_name = row["image_name"].replace(" ", "_")
    
    html += "<tr>"
    html += f"<td><img src='thumbs/{safe_name}'></td>"
    html += f"<td>{row['image_name']}</td>"
    html += f"<td>{row['old_weighted_aesthetic_score']:.4f} (Rank {row['old_weighted_rank']:.0f})</td>"
    html += f"<td>N: {row['nima_unit_score']:.4f}<br>R: {row['rgnet_score']:.4f}<br>A: {row['alamp_score']:.4f}</td>"
    html += f"<td>{row['stage5_student_score']:.4f} (Rank {row['stage5_student_rank']:.0f})</td>"
    html += f"<td>{row['stage5_reference_score']:.4f} (Rank {row['stage5_reference_rank']:.0f})</td>"
    html += f"<td>O-S: {row['rank_delta_old_vs_student']:.0f}<br>S-R: {row['rank_delta_student_vs_reference']:.0f}</td>"
    
    cls_os = row['disagreement_type_old_vs_student']
    cls_sr = row['disagreement_type_student_vs_reference']
    html += f"<td>O-S: <span class='{cls_os}'>{cls_os}</span><br>S-R: <span class='{cls_sr}'>{cls_sr}</span></td>"
    html += f"<td><textarea rows='3' cols='20'></textarea></td>"
    html += "</tr>\n"

html += "</table></body></html>"

with open(os.path.join(OUTPUT_DIR, "review.html"), "w") as f:
    f.write(html)

print("OUTPUT_DIR=" + OUTPUT_DIR)
