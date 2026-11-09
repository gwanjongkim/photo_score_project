import pandas as pd
import numpy as np

# Load data
pred_test = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/predictions_test.csv')

# 1. Dataset Prediction Summary
ds_stats = []
for ds in ['flive', 'koniq10k', 'spaq']:
    subset = pred_test[pred_test['dataset'] == ds]
    target = subset['guarded_target_0_100']
    pred = subset['pred_0_100']
    # Use numeric only for corr
    srcc = subset[['guarded_target_0_100', 'pred_0_100']].corr(method='spearman').iloc[0,1]
    ds_stats.append({
        'dataset': ds,
        'target_mean': target.mean(),
        'target_std': target.std(),
        'pred_mean': pred.mean(),
        'pred_std': pred.std(),
        'bias': pred.mean() - target.mean(),
        'std_ratio': pred.std() / target.std(),
        'mae': (pred - target).abs().mean(),
        'srcc': srcc
    })
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

# 3. Top Abs Errors
pred_test['abs_error'] = (pred_test['pred_0_100'] - pred_test['guarded_target_0_100']).abs()
pred_test.sort_values('abs_error', ascending=False).head(50).to_csv('outputs/mobile_student_iqa_v3_error_analysis_20260601/top_abs_errors.csv', index=False)

# 4. Severe FP Analysis (Filtered to Test and MOS < 40)
df_fp = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv')
test_severe = df_fp[(df_fp['split'] == 'test') & (df_fp['quality_score_0_100'] < 40)].copy()
test_severe['student_amplification'] = test_severe['pred_0_100'] - test_severe['guarded_target_0_100']
test_severe.to_csv('outputs/mobile_student_iqa_v3_error_analysis_20260601/severe_fp_analysis.csv', index=False)
