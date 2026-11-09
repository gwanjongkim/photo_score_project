import pandas as pd
df = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv')
test_df = df[df['split'] == 'test']
severe_test = test_df[test_df['quality_score_0_100'] < 40]
print("Severe FP Dataset Distribution:")
print(severe_test['dataset'].value_counts())
