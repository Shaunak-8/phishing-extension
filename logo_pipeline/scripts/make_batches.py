import pandas as pd
import os
import math

# Settings
INPUT_FILE = "data/processed/urls_labels_raw.csv"
OUTPUT_DIR = "data/batches"
BATCH_SIZE = 500  # you can increase later (1000, 2000, etc.)

def main() -> None:
    """Main function to create batches from input CSV file."""
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")

    # Shuffle for random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(df)
    num_batches = math.ceil(total / BATCH_SIZE)

    print(f"Creating {num_batches} batches of size {BATCH_SIZE}...")

    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)  # Ensure end index does not exceed total rows
        
        batch_df = df.iloc[start:end]
        batch_file = os.path.join(OUTPUT_DIR, f"batch_{i+1}.csv")
        batch_df.to_csv(batch_file, index=False)

        print(f"Saved {batch_file} ({len(batch_df)} rows)")

    print(f"Done. Batches created in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()