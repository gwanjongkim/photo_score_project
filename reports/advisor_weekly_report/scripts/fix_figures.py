import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Output directories
TABLES_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_weekly_report/tables"
FIGURES_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_weekly_report/figures"

model_inventory_data = [
    # Technical
    {'Model': 'TOPIQ-lite (mixed112)', 'Role': 'Technical Quality', 'Dataset': 'SPAQ', 'Input Size': '224x224 (Assumed)', 'Output': 'SRCC 0.899', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': 'Adopted', 'Reason': 'Best Technical Performance', 'Files': 'eval_final_topiq_candidates_vs_existing_technical_20260520'},
    {'Model': 'FLIVE Image Mobile', 'Role': 'Technical Quality', 'Dataset': 'FLIVE', 'Input Size': 'Unknown', 'Output': 'SRCC 0.630', 'TFLite': 'Yes', 'Flutter': 'Unknown', 'Status': 'Hold', 'Reason': 'Lower performance than TOPIQ-lite', 'Files': 'eval_final_topiq_candidates_vs_existing_technical_20260520'},
    {'Model': 'KonIQ Mobile', 'Role': 'Technical Quality', 'Dataset': 'KonIQ', 'Input Size': 'Unknown', 'Output': 'SRCC 0.866', 'TFLite': 'Yes', 'Flutter': 'Unknown', 'Status': 'Hold', 'Reason': 'Lower performance than TOPIQ-lite', 'Files': 'eval_final_topiq_candidates_vs_existing_technical_20260520'},
    {'Model': 'DistortionGuard', 'Role': 'Technical Quality', 'Dataset': 'Synthetic/FLIVE', 'Input Size': 'Unknown', 'Output': 'Bias Issue', 'TFLite': 'No', 'Flutter': 'No', 'Status': 'Failed', 'Reason': 'False positive bias on FLIVE data', 'Files': 'eval_techiqa_guard_v1...'},
    
    # Aesthetic
    {'Model': 'Stage 5 Student', 'Role': 'Aesthetic Quality', 'Dataset': 'AADB', 'Input Size': '224x224', 'Output': 'Acc 0.905', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': 'Discarded', 'Reason': 'Ultimately discarded per project direction', 'Files': 'distill_conservative_aesthetic_full_aadb_20260423'},
    {'Model': 'NIMA (Fixed)', 'Role': 'Aesthetic Quality', 'Dataset': 'AVA', 'Input Size': 'Unknown', 'Output': 'SRCC 0.589', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': 'Adopted', 'Reason': 'Reliable after preproc bug fix', 'Files': 'nima_fixed_preproc_eval_20260525'},
    {'Model': 'RGNet AADB', 'Role': 'Aesthetic Quality', 'Dataset': 'AADB', 'Input Size': 'Unknown', 'Output': 'SRCC 0.681', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': 'Adopted', 'Reason': 'Stable baseline regression', 'Files': 'rgnet_paper_v1_agg_mean_full'},
    {'Model': 'RGNet DenseASPP', 'Role': 'Aesthetic Quality', 'Dataset': 'AADB', 'Input Size': 'Unknown', 'Output': 'SRCC 0.651', 'TFLite': 'No', 'Flutter': 'No', 'Status': 'Failed', 'Reason': 'Did not surpass baseline', 'Files': 'rgnet_v1_cascaded_denseaspp...'},
    {'Model': 'A-LAMP Multi-Patch', 'Role': 'Aesthetic Teacher', 'Dataset': 'AVA', 'Input Size': 'Unknown', 'Output': 'Acc 0.763', 'TFLite': 'Yes', 'Flutter': 'No', 'Status': 'Adopted', 'Reason': 'Used as teacher, not on-device', 'Files': 'alamp_multipatch_teacher_full_ava_20260524'},
    {'Model': 'A-LAMP Graph Fusion', 'Role': 'Aesthetic Quality', 'Dataset': 'AVA (4096)', 'Input Size': 'Unknown', 'Output': 'Overfit', 'TFLite': 'No', 'Flutter': 'No', 'Status': 'Failed', 'Reason': 'Overfitting on small dataset', 'Files': 'alamp_paper_fusion_4096_final_report.md'},
    {'Model': 'ICAA Color', 'Role': 'Color Aesthetic', 'Dataset': 'ICAA17K', 'Input Size': 'Unknown', 'Output': 'SRCC 0.879', 'TFLite': 'Yes', 'Flutter': 'Yes', 'Status': 'Adopted', 'Reason': 'Reproduced paper performance', 'Files': 'icaa_performance_reverification_20260525'},
    
    # Reasoning
    {'Model': 'NVILA-8B', 'Role': 'Explanation', 'Dataset': 'Unknown', 'Input Size': 'Unknown', 'Output': 'Text', 'TFLite': 'No', 'Flutter': 'No', 'Status': 'Hold', 'Reason': 'GPU memory constraints', 'Files': 'acut_product_nvila8b_probe...'},
    {'Model': 'VILA-lite', 'Role': 'Explanation', 'Dataset': 'Unknown', 'Input Size': 'Unknown', 'Output': 'Text', 'TFLite': 'Unknown', 'Flutter': 'Unknown', 'Status': 'Hold', 'Reason': 'Pending further testing', 'Files': 'docs/vila_lite_handoff.md'},
    {'Model': 'Gemma 4-E4B-it', 'Role': 'Explanation', 'Dataset': 'JSON Schema', 'Input Size': 'Unknown', 'Output': 'Text', 'TFLite': 'No', 'Flutter': 'No', 'Status': 'Failed', 'Reason': 'Visual Hallucination', 'Files': 'gemini_api_teacher_aesthetic3_20260427_revised'},
    {'Model': 'Gemini API', 'Role': 'Explanation', 'Dataset': 'N/A', 'Input Size': 'Image', 'Output': 'Text', 'TFLite': 'N/A', 'Flutter': 'Yes', 'Status': 'Adopted', 'Reason': 'Reliable cloud API fallback', 'Files': 'docs/acut_product_integration.md'}
]

# Figure 3: Model Status (English labels to avoid Hangul font warnings)
status_counts = pd.DataFrame(model_inventory_data)['Status'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Model Status Distribution')
plt.savefig(os.path.join(FIGURES_DIR, 'model_status.png'))
plt.close()
