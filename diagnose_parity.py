import pandas as pd
import os
from PIL import Image, ExifTags

df = pd.read_csv('outputs/c6_android_wsl_parity_20260527/android_vs_wsl_parity.csv')

def get_branch(k, t):
    base = 0.7 * t + 0.3 * k
    cap = k + 8.0
    return "cap" if cap <= base else "base"

df['android_branch'] = df.apply(lambda row: get_branch(row['koniq_score_100'], row['topiq_mixed112_score_100']), axis=1)
df['wsl_branch'] = df.apply(lambda row: get_branch(row['wsl_koniq_score_100'], row['wsl_topiq_score_100']), axis=1)
df['branch_match'] = df['android_branch'] == df['wsl_branch']

# Mismatch Attribution
# C6 diff can be explained by KonIQ diff, TOPIQ diff, or Branch switch.
# If branch matches:
#   if base: c6_diff approx 0.7*topiq_diff + 0.3*koniq_diff
#   if cap: c6_diff approx koniq_diff

def attribute(row):
    if not row['branch_match']:
        return "branch_switch"
    if row['android_branch'] == "cap":
        # Contribution is entirely KonIQ
        return "koniq"
    else:
        # Base branch: see which one has higher absolute difference impact
        k_impact = abs(0.3 * row['diff_koniq'])
        t_impact = abs(0.7 * row['diff_topiq'])
        if k_impact > t_impact:
            return "koniq"
        else:
            return "topiq"

df['attribution'] = df.apply(attribute, axis=1)

os.makedirs('outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528', exist_ok=True)
df.to_csv('outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/mismatch_attribution.csv', index=False)

# Top 10 mismatches metadata
top_10 = df.reindex(df.abs_diff_c6.sort_values(ascending=False).index).head(10)

metadata_rows = []
for idx, row in top_10.iterrows():
    filename = row['filename']
    filepath = os.path.join('test_vila', filename)
    meta = {'filename': filename}
    try:
        with Image.open(filepath) as img:
            meta['format'] = img.format
            meta['width'], meta['height'] = img.size
            meta['mode'] = img.mode
            
            exif = img._getexif()
            if exif:
                orientation = None
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == 'Orientation':
                        orientation = value
                        break
                meta['orientation'] = orientation
            else:
                meta['orientation'] = None
            
            meta['icc_profile'] = 'present' if 'icc_profile' in img.info else 'absent'
    except Exception as e:
        meta['error'] = str(e)
    
    meta['abs_diff_c6'] = row['abs_diff_c6']
    meta['attribution'] = row['attribution']
    meta['branch_match'] = row['branch_match']
    metadata_rows.append(meta)

pd.DataFrame(metadata_rows).to_csv('outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/top_mismatch_image_metadata.csv', index=False)

# Branch match stats
print(f"Total images: {len(df)}")
print(f"Branch matches: {df.branch_match.sum()}")
print(f"Branch switches: {len(df) - df.branch_match.sum()}")
print(f"Attribution counts:\n{df.attribution.value_counts()}")

