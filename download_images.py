import pandas as pd
import os
import urllib.request

csv_path = "sample_dataset/images_subset.csv"  #  csv with links
output_folder = "sample_dataset/images"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    filename = row['filename'].strip()
    url = row['link'].strip()
    save_path = os.path.join(output_folder, filename)
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
