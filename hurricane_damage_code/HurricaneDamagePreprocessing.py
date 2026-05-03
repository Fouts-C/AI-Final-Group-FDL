import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image

## PREPROCESSING

#Using Pandas to parse the JSON files and extract the corresponding mask file names
def parse_hurricane_data(json_dir: str, mask_dir: str) -> pd.DataFrame:
    json_path = Path(json_dir)
    mask_path = Path(mask_dir)
    
    parsed_data = []

    # Iterate through all JSON
    for json_file in json_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode JSON in {json_file}")
                continue
            
            # Extract basic frame information
            frame_name = data.get("Frame_Name")
            capture_date = data.get("Capture date")
            region = data.get("Region")
            
            buildings = data.get("Buildings", [])
            for b in buildings:
                #["B004", "26.55438, -77.0968", "1_0373_B0XX_0_Level1.jpg", 1, 1, 1, NaN]
                if len(b) >= 3:
                    building_id = b[0]
                    coordinates = b[1]
                    mask_filename = b[2]
                    
                    label_1 = b[3] if len(b) > 3 else np.nan
                    label_2 = b[4] if len(b) > 4 else np.nan
                    label_3 = b[5] if len(b) > 5 else np.nan
                    label_4 = b[6] if len(b) > 6 else np.nan
                    
                    mask_filepath = mask_path / mask_filename
                    mask_exists = mask_filepath.exists()
                    mask_corrupted = False
                    
                    # Verify if the image file is corrupted
                    if mask_exists:
                        try:
                            with Image.open(mask_filepath) as img:
                                img.verify() # Simple check for corruption
                        except Exception:
                            mask_corrupted = True
                    
                    parsed_data.append({
                        "Frame_Name": frame_name,
                        "Capture_Date": capture_date,
                        "Region": region,
                        "Building_ID": building_id,
                        "Coordinates": coordinates,
                        "Mask_Filename": mask_filename,
                        "Mask_Exists": mask_exists,
                        "Mask_Corrupted": mask_corrupted,
                        "Label_1": label_1,
                        "Label_2": label_2,
                        "Label_3": label_3,
                        "Label_4": label_4
                    })
                    
    df = pd.DataFrame(parsed_data)
    df.replace('NaN', np.nan, inplace=True)
    
    return df

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    # project_root should point to the repository root, not the current directory
    project_root = Path(__file__).parent.parent
    json_directory = project_root / 'data' / 'raw' / 'hurricane_damage' / 'JSON'
    mask_directory = project_root / 'data' / 'raw' / 'hurricane_damage' / 'MASK'
    processed_directory = project_root / 'data' / 'processed'
    processed_directory.mkdir(parents=True, exist_ok=True)
    
    print("Preprocessing Hurricane Damage Data:")
    df_labels = parse_hurricane_data(json_directory, mask_directory)
    print(f"Total raw records parsed: {len(df_labels)}")
    
    # 1. Check for missing masks
    missing_masks = len(df_labels[~df_labels['Mask_Exists']])
    if missing_masks > 0:
        print(f"-> Found {missing_masks} missing mask images.")
    
    # 2. Check for corrupted masks
    corrupted_masks = len(df_labels[df_labels['Mask_Corrupted']])
    if corrupted_masks > 0:
        print(f"-> Found {corrupted_masks} corrupted mask images.")
        
    # 3. Check for missing ground-truth labels (Assuming Label_1 is the primary damage target)
    missing_labels = df_labels['Label_1'].isna().sum()
    if missing_labels > 0:
        print(f"-> Found {missing_labels} records missing the primary target (Label_1).")
    
    # Clean the dataset by dropping bad data
    df_labels = df_labels[
        (df_labels['Mask_Exists'] == True) & 
        (df_labels['Mask_Corrupted'] == False) & 
        (df_labels['Label_1'].notna())
    ]
    
    print(f"Total clean records remaining: {len(df_labels)}")
    
    # 4. Stratified Train/Val/Test Split (70% / 15% / 15%)
    print("\nTrain/val/test split (70/15/15)...")
    
    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df_labels, 
        test_size=0.30, 
        random_state=42, 
        stratify=df_labels['Label_1']
    )
    
    # Second split: Split the 30% Temp into 15% Val and 15% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_df['Label_1']
    )
    
    print(f"Training set:   {len(train_df)} records")
    print(f"Validation set: {len(val_df)} records")
    print(f"Testing set:    {len(test_df)} records")
    
    train_csv = processed_directory / 'hurricane_train_labels.csv'
    val_csv = processed_directory / 'hurricane_val_labels.csv'
    test_csv = processed_directory / 'hurricane_test_labels.csv'
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Saved stratified splits to: {processed_directory}")
