import os
import shutil
import csv

csv_file = "new_labels_clean.csv"
dataset_folder = "dataset_3classes"
output_folder = "dataset_3classes_corrected"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    print("Creating output folder...")
    shutil.copytree(dataset_folder, output_folder)
else:
    print(f"Error: Output folder '{output_folder}' already exists. Please remove or rename it.")

with open(csv_file, "r") as file:
    moved = 0
    reader = csv.reader(file)
    next(reader)  # Skip header row

    print("Moving images...")
    for row in reader:
        print("Row:", row, "out of", len(row))
        index = row[0]
        old_label = row[1]
        new_label = row[2]
        filename = index + ".jpg"
        
        old_path = os.path.join(output_folder, old_label, filename)
        new_path = os.path.join(output_folder, new_label, filename)
        
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"Moved '{filename}' from '{old_label}' to '{new_label}'")
            moved += 1
        else:
            # Check if the image exists in any other label folder
            for label_folder in os.listdir(output_folder):
                if label_folder not in (old_label, new_label):
                    other_label_path = os.path.join(output_folder, label_folder, filename)
                    if os.path.exists(other_label_path):
                        shutil.move(other_label_path, new_path)
                        print(f"Moved '{filename}' from '{label_folder}' to '{new_label}'")
                        moved += 1
                        break
            else:
                print(f"Error: '{filename}' not found in '{old_label}' or other label folders")
    print(f"Moved {moved} images.")
