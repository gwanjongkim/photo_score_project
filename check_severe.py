import pandas as pd
df = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv')
test_df = df[df['split'] == 'test']
severe_test = test_df[test_df['quality_score_0_100'] < 40]
print(f"Total rows in severe_false_positive_cases.csv: {len(df)}")
print(f"Total FPs (MOS < 50) in test: {len(test_df)}")
print(f"Total Severe FPs (MOS < 40) in test: {len(severe_test)}")
