import pandas as pd
import numpy as np
from scipy.stats import spearmanr
df = pd.read_csv("outputs/acut_product_gold_weight_eval_20260603/product_gold_predictions.csv")
w = (0.4, 0.45, 0.15)
scores = df['nima_norm'] * w[0] + df['rgnet_norm'] * w[1] + df['alamp_norm'] * w[2]
print("Scores unique values:", len(scores.unique()))
print("Labels unique values:", len(df['label_ordinal_1_3'].unique()))
srcc, p = spearmanr(df['label_ordinal_1_3'], scores)
print("SRCC:", srcc)
