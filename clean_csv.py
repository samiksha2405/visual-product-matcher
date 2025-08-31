import pandas as pd

# Load the CSV
df = pd.read_csv('sample_dataset/images_subset.csv')

# Keep only the 'filename' and 'link' columns
df_clean = df[['filename', 'link']]

# Drop rows with missing values in either column (optional, but recommended)
df_clean = df_clean.dropna(subset=['filename', 'link'])

# Save the cleaned CSV without extra columns and without the index column
df_clean.to_csv('sample_dataset/images_subset_clean.csv', index=False)

print("Cleaned CSV saved as sample_dataset/images_subset_clean.csv")
