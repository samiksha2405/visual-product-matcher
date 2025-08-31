import os
from PIL import Image

input_folder = 'static/products_raw'
output_folder = 'static/products_resized'
os.makedirs(output_folder, exist_ok=True)
target_size = (224, 224)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            img_resized = img.resize(target_size)
            save_path = os.path.join(output_folder, filename)
            img_resized.save(save_path)
            print(f"Resized and saved {filename}")
