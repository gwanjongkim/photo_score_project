import pandas as pd
df = pd.read_csv('outputs/mobile_student_iqa_v3_20260601/severe_false_positive_cases.csv')
test_df = df[df['split'] == 'test']
severe_test = test_df[test_df['quality_score_0_100'] < 40]

# Check teacher safety for these severe FPs
teacher_fails = severe_test[severe_test['guarded_target_0_100'] > 65]
student_amplified = severe_test[severe_test['guarded_target_0_100'] <= 65]

print(f"Severe FPs in test: {len(severe_test)}")
print(f"Teacher already > 65 (Safe target failure): {len(teacher_fails)}")
print(f"Teacher <= 65, Student > 65 (Student amplification): {len(student_amplified)}")

if len(student_amplified) > 0:
    print("\nStudent amplification examples (Pred - Target):")
    print((student_amplified['pred_0_100'] - student_amplified['guarded_target_0_100']).describe())

