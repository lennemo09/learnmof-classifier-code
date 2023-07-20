import os
import pandas as pd

input_folder = "."
output_file = "new_labels_clean.csv"

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        
        # Remove rows where Original true label is the same as Newly corrected label
        df = df[df["old"] != df["new"]]
        
        # Keep only the first occurrence of each index
        df = df.drop_duplicates(subset=["index"], keep="first")
        
        combined_data = combined_data.append(df, ignore_index=True)

# Write the combined data to the output file
combined_data.to_csv(output_file, index=False)
