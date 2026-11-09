import pandas as pd
import numpy as np
import os

def sample_dataset(df, n, seed=42):
    if len(df) >= n:
        return df.sample(n=n, random_state=seed)
    else:
        print(f"Warning: Dataset size {len(df)} is smaller than requested {n}. Sampling with replacement.")
        return df.sample(n=n, random_state=seed, replace=True)

def prepare_df(path, name):
    df = pd.read_csv(path)
    if 'normalized_mos' not in df.columns:
        df['normalized_mos'] = df['mos'] / 100.0
    
    # Ensure required columns exist
    cols = ['image_path', 'mos', 'normalized_mos', 'dataset', 'is_patch', 'relative_path']
    for col in cols:
        if col not in df.columns:
            if col == 'dataset': df[col] = name
            elif col == 'is_patch': df[col] = 0
            elif col == 'relative_path': df[col] = df['image_path'].apply(lambda x: os.path.basename(x))
            else: df[col] = np.nan
            
    return df[cols]

def main():
    out_dir = 'data/processed/topiq_replacement/'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load source datasets
    spaq_train = prepare_df('data/processed/spaq/train.csv', 'spaq')
    spaq_test = prepare_df('data/processed/spaq/test.csv', 'spaq')
    
    koniq_train = prepare_df('data/processed/koniq10k/splits/train.csv', 'koniq10k')
    koniq_val = prepare_df('data/processed/koniq10k/splits/val.csv', 'koniq10k')
    koniq_test = prepare_df('data/processed/koniq10k/splits/test.csv', 'koniq10k')
    
    flive_train = prepare_df('data/processed/flive/image_train.csv', 'flive_image')
    flive_val = prepare_df('data/processed/flive/image_val.csv', 'flive_image')
    flive_test = prepare_df('data/processed/flive/image_test.csv', 'flive_image')
    
    mixes = [
        {'name': '111', 'ratios': [1, 1, 1]},
        {'name': '112', 'ratios': [1, 1, 2]},
        {'name': '221', 'ratios': [2, 2, 1]}
    ]
    
    unit_train = 4096
    unit_val = 512
    
    summary_rows = []
    
    for mix in mixes:
        m_name = mix['name']
        r = mix['ratios']
        
        # Train
        s_train = sample_dataset(spaq_train, unit_train * r[0])
        k_train = sample_dataset(koniq_train, unit_train * r[1])
        f_train = sample_dataset(flive_train, unit_train * r[2])
        train_df = pd.concat([s_train, k_train, f_train]).sample(frac=1, random_state=42)
        train_df.to_csv(f"{out_dir}mixed_{m_name}_train.csv", index=False)
        
        # Val (Use official val sets if available, otherwise sample from train or test? 
        # Instructions say use val from provided paths)
        s_val = sample_dataset(spaq_test, unit_val * r[0]) # SPAQ has no official val, using test subset
        k_val = sample_dataset(koniq_val, unit_val * r[1])
        f_val = sample_dataset(flive_val, unit_val * r[2])
        val_df = pd.concat([s_val, k_val, f_val]).sample(frac=1, random_state=42)
        val_df.to_csv(f"{out_dir}mixed_{m_name}_val.csv", index=False)
        
        summary_rows.append({
            'Mix': m_name,
            'SPAQ rows': unit_train * r[0],
            'KonIQ rows': unit_train * r[1],
            'FLIVE rows': unit_train * r[2],
            'Total': len(train_df),
            'Notes': f"Ratio {r[0]}:{r[1]}:{r[2]}"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{out_dir}mix_summary.csv", index=False)
    
    with open(f"{out_dir}mix_report.md", 'w') as f:
        f.write("# Mixed Dataset Construction Report\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")

if __name__ == "__main__":
    main()
