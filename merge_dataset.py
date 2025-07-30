import pandas as pd
import os

# Folder where all individual gesture CSVs are saved
input_dir = "gesture_dataset"
output_file = "dataset/A_to_Z_dataset.csv"

# Make sure output directory exists
os.makedirs("dataset", exist_ok=True)

# List all CSV files
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# Merge all into one DataFrame
all_data = []
for file in csv_files:
    path = os.path.join(input_dir, file)
    df = pd.read_csv(path)
    all_data.append(df)
    print(f"📥 Merged: {file}")

# Combine all
merged_df = pd.concat(all_data, ignore_index=True)

# Save merged file
merged_df.to_csv(output_file, index=False)
print(f"\n✅ All files merged into: {output_file}")
