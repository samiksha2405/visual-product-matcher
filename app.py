import os
import csv
import pickle
import numpy as np
import urllib.request
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)


# Configuration
UPLOAD_FOLDER = 'static/uploads'
DATASET_IMG_FOLDER = 'sample_dataset/images'
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'sample_dataset/metadata.csv'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload size


# Load embeddings
with open(EMBEDDINGS_FILE, 'rb') as f:
    embeddings = pickle.load(f)
dataset_ids = list(embeddings.keys())
features = np.array(list(embeddings.values()))


# Load metadata
metadata = {}
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[str(row['id'])] = row


# MobileNetV2 model for feature extraction
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feat = model.predict(img_array, verbose=0).flatten()
        return feat
    except Exception as e:
        print(f"Feature extraction failed for {img_path}: {e}")
        return None


def download_image(url, save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"Failed to download image from URL {url}: {e}")
        return False


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    results = []
    query_image = None

    if request.method == 'POST':
        min_score_str = request.form.get('filter_score')
        min_score = float(min_score_str) / 100 if min_score_str else 0.75
        file = request.files.get('file')
        url = request.form.get('url', '').strip()
        filepath = None

        if file and file.filename != '':
            if not allowed_file(file.filename):
                error = "Unsupported file type. Please upload png/jpg/jpeg/gif."
                return render_template('index.html', error=error)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            query_image = filename

        elif url:
            filename = secure_filename(url.split('/')[-1].split('?')[0])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            if not download_image(url, filepath):
                error = 'Failed to download image from URL.'
                return render_template('index.html', error=error)
            query_image = filename

        else:
            error = 'Please upload a file or enter an image URL.'
            return render_template('index.html', error=error)

        # Extract features and compute similarity
        query_feat = extract_features(filepath)
        if query_feat is None:
            error = "Failed to extract features from the image."
            return render_template('index.html', error=error)
        query_feat = query_feat.reshape(1, -1)

        sim = cosine_similarity(query_feat, features)[0]
        top_idxs = np.argsort(sim)[::-1]

        for i in top_idxs:
            score = sim[i]
            prod_id = dataset_ids[i]
            if score >= min_score:
                row = metadata.get(str(prod_id), {})
                results.append((prod_id, score, row))
            if len(results) >= 10:
                break

        if not results:
            error = f"No matches found with similarity ≥ {min_score*100:.0f}%."

        return render_template('results.html',
                               img_folder=DATASET_IMG_FOLDER,
                               query_image=query_image,
                               results=results,
                               error=error)

    return render_template('index.html')


@app.route('/sample_dataset/images/<filename>')
def serve_product_image(filename):
    return send_from_directory(DATASET_IMG_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
