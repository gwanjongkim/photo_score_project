import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Output directories
TABLES_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_weekly_report/tables"
FIGURES_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_weekly_report/figures"

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Timeline Table
timeline_data = [
    {'Week': 'Week 1', 'Date': '2026-03-30 ~ 2026-04-05', 'Event': 'Stage 5 Smoke Test & Firebase Worker Local Smoke'},
    {'Week': 'Week 2', 'Date': '2026-04-06 ~ 2026-04-12', 'Event': 'Backend Cleanup & Stage 5 Distillation (AADB subset)'},
    {'Week': 'Week 3', 'Date': '2026-04-13 ~ 2026-04-19', 'Event': 'NVILA-8B Inference Probe & VILA-lite Handoff'},
    {'Week': 'Week 4', 'Date': '2026-04-20 ~ 2026-04-26', 'Event': 'Stage 5 Student Final Evaluation (Discarded later)'},
    {'Week': 'Week 5', 'Date': '2026-04-27 ~ 2026-05-03', 'Event': 'Gemma 4-E4B-it QLoRA Adapter Training'},
    {'Week': 'Week 6', 'Date': '2026-05-04 ~ 2026-05-10', 'Event': 'RGNet & A-LAMP Baseline Audits'},
    {'Week': 'Week 7', 'Date': '2026-05-11 ~ 2026-05-17', 'Event': 'A-LAMP Graph Fusion & ICAA TF Native Conversion'},
    {'Week': 'Week 8', 'Date': '2026-05-18 ~ 2026-05-24', 'Event': 'TOPIQ-lite Benchmark & DistortionGuard Development'},
    {'Week': 'Week 9', 'Date': '2026-05-25 ~ 2026-05-31', 'Event': 'NIMA Preprocessing Bug Fix & Final TFLite Pack'}
]
pd.DataFrame(timeline_data).to_csv(os.path.join(TABLES_DIR, 'experiment_timeline.csv'), index=False)

# 2. Model Inventory Table
# Adding explicit discard for Stage 5 based on instruction
model_inventory_data = [
    # Technical
    {'Model': 'TOPIQ-lite (mixed112)', 'Role': 'Technical Quality', 'Dataset': 'SPAQ', 'Input Size': '224x224 (Assumed)', 'Output': 'SRCC 0.899', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': '사용', 'Reason': 'Best Technical Performance', 'Files': 'eval_final_topiq_candidates_vs_existing_technical_20260520'},
    {'Model': 'FLIVE Image Mobile', 'Role': 'Technical Quality', 'Dataset': 'FLIVE', 'Input Size': 'Unknown', 'Output': 'SRCC 0.630', 'TFLite': 'Yes', 'Flutter': 'Unknown', 'Status': '보류', 'Reason': 'Lower performance than TOPIQ-lite', 'Files': 'eval_final_topiq_candidates_vs_existing_technical_20260520'},
    {'Model': 'KonIQ Mobile', 'Role': 'Technical Quality', 'Dataset': 'KonIQ', 'Input Size': 'Unknown', 'Output': 'SRCC 0.866', 'TFLite': 'Yes', 'Flutter': 'Unknown', 'Status': '보류', 'Reason': 'Lower performance than TOPIQ-lite', 'Files': 'eval_final_topiq_candidates_vs_existing_technical_20260520'},
    {'Model': 'DistortionGuard', 'Role': 'Technical Quality', 'Dataset': 'Synthetic/FLIVE', 'Input Size': 'Unknown', 'Output': 'Bias Issue', 'TFLite': 'No', 'Flutter': 'No', 'Status': '실패', 'Reason': 'False positive bias on FLIVE data', 'Files': 'eval_techiqa_guard_v1...'},
    
    # Aesthetic
    {'Model': 'Stage 5 Student', 'Role': 'Aesthetic Quality', 'Dataset': 'AADB', 'Input Size': '224x224', 'Output': 'Acc 0.905', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': '사용 안 함', 'Reason': 'Ultimately discarded per project direction', 'Files': 'distill_conservative_aesthetic_full_aadb_20260423'},
    {'Model': 'NIMA (Fixed)', 'Role': 'Aesthetic Quality', 'Dataset': 'AVA', 'Input Size': 'Unknown', 'Output': 'SRCC 0.589', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': '사용', 'Reason': 'Reliable after preproc bug fix', 'Files': 'nima_fixed_preproc_eval_20260525'},
    {'Model': 'RGNet AADB', 'Role': 'Aesthetic Quality', 'Dataset': 'AADB', 'Input Size': 'Unknown', 'Output': 'SRCC 0.681', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': '사용', 'Reason': 'Stable baseline regression', 'Files': 'rgnet_paper_v1_agg_mean_full'},
    {'Model': 'RGNet DenseASPP', 'Role': 'Aesthetic Quality', 'Dataset': 'AADB', 'Input Size': 'Unknown', 'Output': 'SRCC 0.651', 'TFLite': 'No', 'Flutter': 'No', 'Status': '실패', 'Reason': 'Did not surpass baseline', 'Files': 'rgnet_v1_cascaded_denseaspp...'},
    {'Model': 'A-LAMP Multi-Patch', 'Role': 'Aesthetic Teacher', 'Dataset': 'AVA', 'Input Size': 'Unknown', 'Output': 'Acc 0.763', 'TFLite': 'Yes', 'Flutter': 'No', 'Status': '사용', 'Reason': 'Used as teacher, not on-device', 'Files': 'alamp_multipatch_teacher_full_ava_20260524'},
    {'Model': 'A-LAMP Graph Fusion', 'Role': 'Aesthetic Quality', 'Dataset': 'AVA (4096)', 'Input Size': 'Unknown', 'Output': 'Overfit', 'TFLite': 'No', 'Flutter': 'No', 'Status': '실패', 'Reason': 'Overfitting on small dataset', 'Files': 'alamp_paper_fusion_4096_final_report.md'},
    {'Model': 'ICAA Color', 'Role': 'Color Aesthetic', 'Dataset': 'ICAA17K', 'Input Size': 'Unknown', 'Output': 'SRCC 0.879', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': '사용', 'Reason': 'Reproduced paper performance', 'Files': 'icaa_performance_reverification_20260525'},
    
    # Reasoning
    {'Model': 'NVILA-8B', 'Role': 'Explanation', 'Dataset': 'Unknown', 'Input Size': 'Unknown', 'Output': 'Text', 'TFLite': 'No', 'Flutter': 'No', 'Status': '보류', 'Reason': 'GPU memory constraints', 'Files': 'acut_product_nvila8b_probe...'},
    {'Model': 'VILA-lite', 'Role': 'Explanation', 'Dataset': 'Unknown', 'Input Size': 'Unknown', 'Output': 'Text', 'TFLite': 'Unknown', 'Flutter': 'Unknown', 'Status': '보류', 'Reason': 'Pending further testing', 'Files': 'docs/vila_lite_handoff.md'},
    {'Model': 'Gemma 4-E4B-it', 'Role': 'Explanation', 'Dataset': 'JSON Schema', 'Input Size': 'Unknown', 'Output': 'Text', 'TFLite': 'No', 'Flutter': 'No', 'Status': '실패', 'Reason': 'Visual Hallucination', 'Files': 'gemini_api_teacher_aesthetic3_20260427_revised'},
    {'Model': 'Gemini API', 'Role': 'Explanation', 'Dataset': 'N/A', 'Input Size': 'Image', 'Output': 'Text', 'TFLite': 'N/A', 'Flutter': 'Yes', 'Status': '사용', 'Reason': 'Reliable cloud API fallback', 'Files': 'docs/acut_product_integration.md'}
]
pd.DataFrame(model_inventory_data).to_csv(os.path.join(TABLES_DIR, 'model_inventory.csv'), index=False)

# 3. Success/Failure Matrix
matrix_data = [
    {'Experiment': 'Stage 5 Distillation', 'Goal': 'Mobile Ensemble', 'Result': 'Acc 0.905, 1.7MB', 'Status': '폐기', 'Reason': 'Decided not to use Stage 5 in final product', 'Files': 'distill_conservative_aesthetic...'},
    {'Experiment': 'Gemma Adapter Training', 'Goal': 'On-device JSON generation', 'Result': 'Hallucination', 'Status': '실패', 'Reason': 'Visual hallucination makes it unusable', 'Files': 'gemini_api_teacher_aesthetic3_...'},
    {'Experiment': 'A-LAMP Graph Fusion', 'Goal': 'Layout-aware aesthetic score', 'Result': 'Overfit', 'Status': '실패', 'Reason': 'Overfitted on 4k dataset', 'Files': 'alamp_paper_fusion_4096_final_report.md'},
    {'Experiment': 'RGNet DenseASPP', 'Goal': 'Improve RGNet baseline', 'Result': 'SRCC 0.651', 'Status': '실패', 'Reason': 'Worse than mean-agg baseline (0.681)', 'Files': 'rgnet_v1_cascaded_denseaspp...'},
    {'Experiment': 'DistortionGuard', 'Goal': 'Technical flaw detection', 'Result': 'Bias on FLIVE', 'Status': '실패', 'Reason': 'High false positives', 'Files': 'eval_techiqa_guard...'},
    {'Experiment': 'NIMA Preproc Fix', 'Goal': 'Fix double scaling bug', 'Result': 'SRCC 0.589', 'Status': '성공', 'Reason': 'Significant performance bump', 'Files': 'nima_fixed_preproc_eval_20260525'},
    {'Experiment': 'TOPIQ-lite Benchmark', 'Goal': 'Technical quality scoring', 'Result': 'SRCC 0.899', 'Status': '성공', 'Reason': 'Outperformed FLIVE/KonIQ', 'Files': 'eval_final_topiq_candidates...'},
    {'Experiment': 'ICAA TF Native', 'Goal': 'Color aesthetics TFLite', 'Result': 'SRCC 0.879', 'Status': '성공', 'Reason': 'Parity with PyTorch', 'Files': 'icaa_performance_reverification_20260525'},
]
pd.DataFrame(matrix_data).to_csv(os.path.join(TABLES_DIR, 'success_failure_matrix.csv'), index=False)

# 4. Generate Figures
# Figure 2: Timeline
plt.figure(figsize=(10, 5))
weeks = [d['Week'] for d in timeline_data]
y_pos = np.arange(len(weeks))
plt.barh(y_pos, [1]*len(weeks), align='center', color='lightblue', edgecolor='black')
plt.yticks(y_pos, weeks)
plt.xlabel('Progress')
plt.title('Project Timeline by Week')
for i, d in enumerate(timeline_data):
    plt.text(0.05, i, d['Event'], va='center', fontsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'timeline.png'))
plt.close()

# Figure 3: Model Status
status_counts = pd.DataFrame(model_inventory_data)['Status'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Model Status Distribution')
plt.savefig(os.path.join(FIGURES_DIR, 'model_status.png'))
plt.close()

# Figure 4: Performance Comparison (SRCC only)
srcc_models = [m for m in model_inventory_data if 'SRCC' in str(m['Output'])]
names = [m['Model'] for m in srcc_models]
scores = [float(m['Output'].replace('SRCC ', '')) for m in srcc_models]
plt.figure(figsize=(10, 5))
plt.bar(names, scores, color='lightgreen', edgecolor='black')
plt.ylabel('SRCC')
plt.title('Performance Comparison (SRCC)')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, str(v), ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'performance_srcc.png'))
plt.close()

print("Files generated successfully.")
