import os
import pickle
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tqdm import tqdm
import pandas as pd

IMG_FOLDER = "sample_dataset/images"
METADATA_FILE = "sample_dataset/metadata.csv"


model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

embeddings = {}
df = pd.read_csv(METADATA_FILE)
ids = df['id'].astype(str).tolist()

for prod_id in tqdm(ids):
    img_name = f"{prod_id}.jpg"
    img_path = os.path.join(IMG_FOLDER, img_name)
    if not os.path.exists(img_path):
        print(f"Missing image: {img_name}")
        continue
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feat = model.predict(img_array, verbose=0).flatten()
        embeddings[prod_id] = feat
    except Exception as e:
        print(f"Error: {img_name} {e}")

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
print(f"Saved {len(embeddings)} embeddings.")
