import pandas as pd

print("=== AVA Dataset ===")
splits = ['train.csv', 'val.csv', 'test.csv']
for s in splits:
    try:
        df = pd.read_csv(f'data/processed/ava/{s}')
        print(f"Split: {s}")
        print("Columns:", list(df.columns))
        print("Count:", len(df))
        if 'mean_score' in df.columns:
            print("mean_score min/max/mean:", df['mean_score'].min(), df['mean_score'].max(), df['mean_score'].mean())
        elif 'mean' in df.columns:
            print("mean min/max/mean:", df['mean'].min(), df['mean'].max(), df['mean'].mean())
    except Exception as e:
        print(e)

print("\n=== AADB Dataset ===")
splits = ['train.csv', 'val.csv', 'test.csv']
for s in splits:
    try:
        df = pd.read_csv(f'data/processed/aadb/{s}')
        print(f"Split: {s}")
        print("Columns:", list(df.columns))
        print("Count:", len(df))
        if 'score' in df.columns:
            print("score min/max/mean:", df['score'].min(), df['score'].max(), df['score'].mean())
    except Exception as e:
        print(e)
