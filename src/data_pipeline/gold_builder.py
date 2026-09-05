import os
import json
import shutil
import pandas as pd

SILVER_FILE = '/app/data/silver/cleaned_publications.json'
GOLD_DIR = '/app/data/gold/publications'

def build_gold_lakehouse():
    """Converts Silver JSON publications layer into a partitioned Gold Parquet Lakehouse."""
    if not os.path.exists(SILVER_FILE):
        print(f"[WARNING] Silver file not found at: {SILVER_FILE}")
        return

    try:
        with open(SILVER_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            print("[WARNING] Silver dataset is empty. Aborting Gold build.")
            return

        df = pd.DataFrame(data)

        if 'publication_year' not in df.columns:
            print("[ERROR] Column 'publication_year' missing from Silver data.")
            return

        initial_len = len(df)
        df = df.dropna(subset=['publication_year']).copy()
        df['publication_year'] = df['publication_year'].astype(int)

        # Convert nested list/dict columns to JSON strings for PyArrow compatibility
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)

        if os.path.exists(GOLD_DIR):
            shutil.rmtree(GOLD_DIR)
        os.makedirs(GOLD_DIR, exist_ok=True)

        df.to_parquet(
            GOLD_DIR,
            partition_cols=['publication_year'],
            index=False,
            engine='pyarrow'
        )

        partitions = [d for d in os.listdir(GOLD_DIR) if d.startswith('publication_year=')]
        print(f"[SUCCESS] Partitioned Gold Lakehouse created successfully!")
        print(f"[INFO] Processed {len(df)}/{initial_len} records across {len(partitions)} year partitions in '{GOLD_DIR}'.")

    except Exception as e:
        print(f"[ERROR] Failed to build Gold Parquet Lakehouse: {str(e)}")

if __name__ == '__main__':
    build_gold_lakehouse()
