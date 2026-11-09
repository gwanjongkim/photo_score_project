import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Input files
METRICS_CSV = "/home/omen_pc1/photo_score_project/outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv"
RGNET_PREDS = "/home/omen_pc1/photo_score_project/outputs/rgnet_paper_v1_aadb_regression_20260509/eval/test_predictions.csv"

# Output directories
FIGURES_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_report/figures"
TABLES_DIR = "/home/omen_pc1/photo_score_project/reports/advisor_report/tables"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_technical_chart(df):
    tech_df = df[df['category'] == 'Technical'].copy()
    if tech_df.empty:
        return
    
    # We want to plot SRCC for technical models
    tech_df = tech_df.dropna(subset=['srcc'])
    tech_df = tech_df.sort_values(by='srcc', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tech_df['model_name'] + "\\n(" + tech_df['dataset'] + ")", tech_df['srcc'], color='skyblue')
    plt.title('C. Technical Quality Models Comparison (SRCC)', fontsize=14)
    plt.ylabel('SRCC (Higher is better)', fontsize=12)
    plt.xlabel('Model & Dataset', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'C_technical_quality_comparison.png'))
    plt.close()

def generate_aesthetic_chart(df):
    aes_df = df[df['category'] == 'Aesthetic'].copy()
    if aes_df.empty:
        return
    
    # We have mixed metrics (SRCC and Accuracy). We will create a grouped bar chart or two subplots.
    srcc_df = aes_df[aes_df['metric_primary_name'] == 'SRCC'].copy()
    acc_df = aes_df[aes_df['metric_primary_name'].str.contains('Accuracy')].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # SRCC Plot
    if not srcc_df.empty:
        srcc_df = srcc_df.sort_values(by='metric_primary_value', ascending=False)
        bars1 = ax1.bar(srcc_df['model_name'] + "\\n(" + srcc_df['dataset'] + ")", srcc_df['metric_primary_value'], color='lightcoral')
        ax1.set_title('Aesthetic Models Comparison (SRCC)', fontsize=12)
        ax1.set_ylabel('SRCC', fontsize=10)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, round(bar.get_height(), 3), ha='center', va='bottom')

    # Accuracy Plot
    if not acc_df.empty:
        acc_df = acc_df.sort_values(by='metric_primary_value', ascending=False)
        bars2 = ax2.bar(acc_df['model_name'] + "\\n(" + acc_df['dataset'] + ")", acc_df['metric_primary_value'], color='lightgreen')
        ax2.set_title('Aesthetic Models Comparison (Accuracy)', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, round(bar.get_height(), 3), ha='center', va='bottom')

    plt.suptitle('D. Aesthetic Quality Models Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'D_aesthetic_quality_comparison.png'))
    plt.close()

def generate_distribution_histogram(preds_df):
    plt.figure(figsize=(8, 5))
    plt.hist(preds_df['score'], bins=20, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
    plt.hist(preds_df['prediction'], bins=20, alpha=0.5, label='Prediction', color='orange', edgecolor='black')
    plt.title('E. Score Distribution Histogram (RGNet on AADB)', fontsize=14)
    plt.xlabel('Aesthetic Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'E_score_distribution.png'))
    plt.close()

def generate_scatter_plot(preds_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(preds_df['score'], preds_df['prediction'], alpha=0.5, color='purple', s=20)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Ideal (y=x)')
    
    # Calculate SRCC and PLCC
    srcc = preds_df['score'].corr(preds_df['prediction'], method='spearman')
    plcc = preds_df['score'].corr(preds_df['prediction'], method='pearson')
    
    plt.title('F. Prediction Scatter Plot (RGNet on AADB)', fontsize=14)
    plt.xlabel('Ground Truth Score', fontsize=12)
    plt.ylabel('Predicted Score', fontsize=12)
    plt.legend()
    plt.text(0.1, 0.9, f'SRCC: {srcc:.3f}\\nPLCC: {plcc:.3f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'F_prediction_scatter.png'))
    plt.close()

def generate_hard_case_grid(preds_df):
    # Find top 2 highest over-predicted (hard false positive) and top 2 highest under-predicted
    preds_df['error'] = preds_df['prediction'] - preds_df['score']
    over_preds = preds_df.nlargest(2, 'error')
    under_preds = preds_df.nsmallest(2, 'error')
    
    hard_cases = pd.concat([over_preds, under_preds])
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(hard_cases.iterrows()):
        img_path = row['image_path']
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                title_color = 'red' if row['error'] > 0 else 'blue'
                axes[i].set_title(f"GT: {row['score']:.2f} | Pred: {row['prediction']:.2f}\\nError: {row['error']:+.2f}", color=title_color)
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Image Load Error", ha='center', va='center')
        else:
            axes[i].text(0.5, 0.5, "Image File Not Found", ha='center', va='center')
        axes[i].axis('off')
        
    plt.suptitle('G. Hard Case Image Grid (Over-predicted vs Under-predicted)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'G_hard_case_grid.png'))
    plt.close()

def save_tables(df):
    # B. Model summary table
    summary_cols = ['model_name', 'category', 'dataset', 'tflite_exported', 'notes']
    summary_df = df[summary_cols].copy()
    summary_df.to_csv(os.path.join(TABLES_DIR, 'B_model_summary.csv'), index=False)
    
    # I. Decision Matrix table
    # Adding mock heuristic decision columns based on report
    decision_data = []
    for _, row in df.iterrows():
        decision = "Adopted" if "true" in str(row['tflite_exported']).lower() else "Hold"
        if "Failed" in str(row['notes']):
            decision = "Discarded"
        
        decision_data.append({
            'Model': row['model_name'],
            'Role': row['category'],
            'Metric': f"{row['metric_primary_name']} {row['metric_primary_value']}",
            'Mobile Ready': row['mobile_ready'],
            'Implementation Diff': 'Medium',
            'Decision': decision,
            'Reason': row['notes']
        })
    pd.DataFrame(decision_data).to_csv(os.path.join(TABLES_DIR, 'I_decision_matrix.csv'), index=False)

if __name__ == "__main__":
    print("Loading data...")
    metrics_df = pd.read_csv(METRICS_CSV)
    preds_df = pd.read_csv(RGNET_PREDS)
    
    print("Generating figures...")
    generate_technical_chart(metrics_df)
    generate_aesthetic_chart(metrics_df)
    generate_distribution_histogram(preds_df)
    generate_scatter_plot(preds_df)
    generate_hard_case_grid(preds_df)
    
    print("Generating tables...")
    save_tables(metrics_df)
    print("Done!")
