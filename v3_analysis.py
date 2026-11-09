import pandas as pd
import numpy as np
import os

# Create output dir
os.makedirs('outputs/mobile_student_iqa_v3_error_analysis_20260601', exist_ok=True)

# Load data
pred_train = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/predictions_train.csv')
pred_val = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/predictions_val.csv')
pred_test = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/predictions_test.csv')
severe_fp_cases = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv')

def get_stats(df, name):
    target = df['guarded_target_0_100']
    pred = df['pred_0_100']
    return {
        'split': name,
        'target_mean': target.mean(),
        'target_std': target.std(),
        'pred_mean': pred.mean(),
        'pred_std': pred.std(),
        'bias': pred.mean() - target.mean(),
        'compression': pred.std() / target.std() if target.std() > 0 else 0,
        'mae': (pred - target).abs().mean()
    }

# 1. Dataset Prediction Summary
ds_stats = []
for ds in pred_test['dataset'].unique():
    subset = pred_test[pred_test['dataset'] == ds]
    s = get_stats(subset, ds)
    ds_stats.append(s)
pd.DataFrame(ds_stats).to_csv('outputs/mobile_student_iqa_v3_error_analysis_20260601/dataset_prediction_summary.csv', index=False)

# 2. Decile Error Summary
pred_test['target_decile'] = pd.qcut(pred_test['guarded_target_0_100'], 10, labels=False, duplicates='drop')
decile_summary = pred_test.groupby('target_decile').agg({
    'guarded_target_0_100': ['mean', 'min', 'max'],
    'pred_0_100': ['mean', 'std'],
}).reset_index()
decile_summary.columns = ['decile', 'target_mean', 'target_min', 'target_max', 'pred_mean', 'pred_std']
decile_summary['mae'] = pred_test.groupby('target_decile').apply(lambda x: (x['pred_0_100'] - x['guarded_target_0_100']).abs().mean(), include_groups=False).values
decile_summary.to_csv('outputs/mobile_student_iqa_v3_error_analysis_20260601/decile_error_summary.csv', index=False)

# 3. Top Absolute Errors
pred_test['abs_error'] = (pred_test['pred_0_100'] - pred_test['guarded_target_0_100']).abs()
pred_test.sort_values('abs_error', ascending=False).head(100).to_csv('outputs/mobile_student_iqa_v3_error_analysis_20260601/top_abs_errors.csv', index=False)

# 4. Severe FP Analysis
# quality_score_0_100 < 40 and pred_0_100 > 65
# Target-inherited: guarded_target_0_100 > 65 (Teacher already failed)
# Student-amplified: guarded_target_0_100 <= 65 but pred_0_100 > 65
severe_fp_analysis = severe_fp_cases.copy()
severe_fp_analysis['category'] = np.where(
    severe_fp_analysis['guarded_target_0_100'] > 65, 'target-inherited',
    np.where(severe_fp_analysis['guarded_target_0_100'] <= 55, 'student-amplified-severe', 'mixed')
)
severe_fp_analysis['student_amplification'] = severe_fp_analysis['pred_0_100'] - severe_fp_analysis['guarded_target_0_100']
severe_fp_analysis.to_csv('outputs/mobile_student_iqa_v3_error_analysis_20260601/severe_fp_analysis.csv', index=False)

# Print some results for the thought process
print("Dataset Prediction Summary:")
print(pd.DataFrame(ds_stats))
print("\nSevere FP Categories:")
print(severe_fp_analysis['category'].value_counts())
