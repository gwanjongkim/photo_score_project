import pandas as pd
df = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/predictions_test.csv')

def summarize_ds(ds_name):
    subset = df[df['dataset'] == ds_name].copy()
    subset['target_decile'] = pd.qcut(subset['guarded_target_0_100'], 10, labels=False, duplicates='drop')
    summary = subset.groupby('target_decile').agg({
        'guarded_target_0_100': ['mean', 'count'],
        'pred_0_100': 'mean'
    }).reset_index()
    summary.columns = ['decile', 'target_mean', 'count', 'pred_mean']
    summary['bias'] = summary['pred_mean'] - summary['target_mean']
    print(f"\nDecile Bias for {ds_name}:")
    print(summary)

for ds in ['spaq', 'koniq10k', 'flive']:
    summarize_ds(ds)
