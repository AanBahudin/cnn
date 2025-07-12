from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os
import random

# ======= Konfigurasi =======
class_names = [
    'Bancana Kaluku Bula Laki laki', 
    'Bancana Kaluku Bula Perempuan', 
    'Katamba Gawu Laki laki', 
    'Katamba Gawu Perempuan',
    'Manggopa Laki laki',
    'Manggopa Perempuan',
    'Samasili Laki laki',
    'Samasili Perempuan'
]

DATASET_DIR = "dataset/ALL"  # Lokasi dataset
MODEL_PATH = "models/v7.2/model_cnn.h5"
# MODEL_PATH = "model_cnn.h5"

# ======= Inisialisasi Blueprint =======
predict_blueprint = Blueprint('predict', __name__)

# ======= Fungsi Center Crop =======
def center_crop_and_resize(img, target_size=(224, 224)):
    width, height = img.size
    crop_size = int(min(width, height) * 0.875)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img

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

# ======= Endpoint Prediksi =======
@predict_blueprint.route('/data/processing', methods=['POST'])
def process_post():
    model = tf.keras.models.load_model(MODEL_PATH)

    if 'motifKain' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['motifKain']

    try:
        # Baca gambar dan preprocessing
        img = Image.open(image_file.stream).convert("RGB")
        img = center_crop_and_resize(img, (224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        # Ambil 3 contoh gambar dari label yang diprediksi
        example_images = get_random_images_by_label(predicted_label)

        return jsonify({
            'label': predicted_label,
            'confidence': f"{confidence:.2%}",
            'confidence_raw': confidence,
            'examples': example_images
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
