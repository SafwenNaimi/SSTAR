

import os
import json
import numpy as np

main_folder_path = 'C:/Users/safwe/Documents/Coding/AcT_STM/Action_keypoint+angles/walk'
output_folder_path = 'C:/Users/safwe/Documents/Coding/AcT_STM/Action_keypoint+angles/walk'

for root, dirs, files in os.walk(main_folder_path):
    for folder_name in dirs:
        folder_path = os.path.join(root, folder_name)
        output_file_path = os.path.join(output_folder_path, folder_name)
        
        # Initialize an empty list to store the data
        all_data = []

        # Iterate over each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    # Load JSON data from the file
                    data = json.load(f)
                    # Append the data to the list
                    all_data.append(data)

        # Convert the list of data into a numpy array
        concatenated_data = np.concatenate(all_data)

        # Save the data with the folder name
        #np.save(output_file_path, concatenated_data)
        print(f" with shape {concatenated_data.shape}")

