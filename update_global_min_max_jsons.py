#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:40:55 2026

@author: wirrbel
"""

import os
import json

def update_json_files(folder_path):
    # The data you want to add
    updates = {
        "global_min": 300, 
        "global_max": 5000
    }

    # Iterate over files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 1. Read the existing JSON data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 2. Update the data with new values
                # This works if the JSON root is an object/dictionary
                if isinstance(data, dict):
                    data.update(updates)
                    
                    # 3. Write the updated data back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                    print(f"Updated: {filename}")
                else:
                    print(f"Skipped {filename}: Root is not a JSON object")
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing {filename}: {e}")

# Use '.' for the current folder, or provide a full path like 'C:/my_folder'
folder_to_process = '/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_raw/Dataset020_wfa_moritz_ilastik/labelsTr/' 
update_json_files(folder_to_process)