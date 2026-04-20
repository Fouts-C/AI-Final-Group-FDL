import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

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
                    
                    parsed_data.append({
                        "Frame_Name": frame_name,
                        "Capture_Date": capture_date,
                        "Region": region,
                        "Building_ID": building_id,
                        "Coordinates": coordinates,
                        "Mask_Filename": mask_filename,
                        "Mask_Exists": mask_exists,
                        "Label_1": label_1,
                        "Label_2": label_2,
                        "Label_3": label_3,
                        "Label_4": label_4
                    })
                    
    df = pd.DataFrame(parsed_data)
    df.replace('NaN', np.nan, inplace=True)
    
    return df

if __name__ == "__main__":
    project_root = Path(__file__).parent
    json_directory = project_root / 'data' / 'raw' / 'hurricane_damage' / 'JSON'
    mask_directory = project_root / 'data' / 'raw' / 'hurricane_damage' / 'MASK'
    processed_directory = project_root / 'data' / 'processed'
    processed_directory.mkdir(parents=True, exist_ok=True)
    
    df_labels = parse_hurricane_data(json_directory, mask_directory)
    
    missing_masks = len(df_labels[~df_labels['Mask_Exists']])
    
    df_labels = df_labels[df_labels['Mask_Exists']]
    
    output_csv = processed_directory / 'hurricane_damage_labels.csv'
    df_labels.to_csv(output_csv, index=False)
