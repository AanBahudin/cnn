from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import base64

# === Konfigurasi ===
DATASET_DIR = "dataset/ALL"
MODEL_PATH = "models/v7.2/model_cnn.h5"
CENTROID_PATH = "centroids.pkl"
IMAGE_SIZE = (224, 224)
SIMILARITY_THRESHOLD = 0.75

predict_blueprint = Blueprint('predict', __name__)

# Load model dan feature extractor
base_model = load_model(MODEL_PATH)
feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

# Load centroid data
with open(CENTROID_PATH, "rb") as f:
    data = pickle.load(f)
    centroids = data["centroids"]
    class_names = data["class_names"]

# Preprocessing
def preprocess_image_file(file):
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        raise ValueError(f"[ERROR] Gagal membuka gambar: {e}")
    width, height = img.size
    crop_size = int(min(width, height) * 0.875)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======= Ambil 3 Gambar Acak (Base64) =======
def get_random_images_by_label(label, dataset_dir=DATASET_DIR, count=5):
    label_dir = os.path.join(dataset_dir, label)
    if not os.path.exists(label_dir):
        return []

    all_images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(all_images, min(count, len(all_images)))

    base64_images = []
    for img_name in selected:
        img_path = os.path.join(label_dir, img_name)
        with open(img_path, 'rb') as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            base64_images.append(f"data:image/jpeg;base64,{encoded}")
    return base64_images


# Endpoint prediksi dengan OOD
@predict_blueprint.route('/data/processing', methods=['POST'])
def predict():
    if 'motifKain' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        img_tensor = preprocess_image_file(request.files['motifKain'])
        feature = feature_extractor.predict(img_tensor, verbose=0)[0]

        similarities = {
            class_name: cosine_similarity([feature], [centroid])[0][0]
            for class_name, centroid in centroids.items()
        }

        best_label = max(similarities, key=similarities.get)
        best_score = similarities[best_label]

        example_images = get_random_images_by_label(best_label)

        if best_score < SIMILARITY_THRESHOLD:
            return jsonify({
                "status": "unknown",
                "message": "Gambar tidak dikenali sebagai salah satu dari 8 motif tenun",
                "score": float(best_score)
            }), 200
        else:
            return jsonify({
                "status": "recognized",
                "label": best_label,
                "score": float(best_score),
                "examples": example_images
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
