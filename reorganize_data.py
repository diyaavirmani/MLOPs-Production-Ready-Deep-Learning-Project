import os
import pandas as pd
import shutil

def reorganize(split):
    base_dir = f"artifacts/data_ingestion/{split}"
    csv_path = os.path.join(base_dir, "_classes.csv")
    
    if not os.path.exists(csv_path):
        print(f"No CSV found for {split}")
        return

    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Target classes: Normal and Cancer (everything else)
    # If the user specified 2 classes, this is a reasonable split.
    # Alternatively, we could just use the columns as is, but params.yaml says 2.
    
    for _, row in df.iterrows():
        filename = row['filename']
        src_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(src_path):
            continue
            
        # Check if normal (column 'normal')
        if row['normal'] == 1:
            class_folder = "normal"
        else:
            class_folder = "cancer"
            
        dest_dir = os.path.join(base_dir, class_folder)
        os.makedirs(dest_dir, exist_ok=True)
        
        shutil.move(src_path, os.path.join(dest_dir, filename))

if __name__ == "__main__":
    reorganize("train")
    reorganize("valid")
    reorganize("test")
    print("Data reorganization complete.")
