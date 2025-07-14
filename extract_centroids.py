import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from PIL import Image
import pickle

# === Konfigurasi ===
DATASET_DIR = "dataset/TRAINING"
MODEL_PATH = "models/v7.3/model_cnn.h5"
CENTROID_SAVE_PATH = "centroids.pkl"
IMAGE_SIZE = (224, 224)

# Fungsi preprocessing: center crop & resize
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Tidak bisa membuka gambar {image_path}: {e}")
        return None

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

# Load model Sequential yang sudah dilatih
model = load_model(MODEL_PATH)

# Buat feature extractor (tanpa layer softmax terakhir)
input_tensor = Input(shape=(224, 224, 3))
x = input_tensor
for layer in model.layers[:-1]:  # Hilangkan layer softmax terakhir
    x = layer(x)
feature_extractor = Model(inputs=input_tensor, outputs=x)

# Ambil nama kelas dari direktori
class_names = sorted(os.listdir(DATASET_DIR))

# Ekstrak fitur dan hitung centroid untuk tiap kelas
centroids = {}
for class_name in class_names:
    class_dir = os.path.join(DATASET_DIR, class_name)
    print(f"\n🔍 Memproses kelas: {class_name}")
    features = []

    for file in os.listdir(class_dir):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(class_dir, file)
        img_tensor = preprocess_image(img_path)
        if img_tensor is None:
            continue
        try:
            feature = feature_extractor.predict(img_tensor, verbose=0)[0]
            features.append(feature)
        except Exception as e:
            print(f"[SKIP] Gagal memproses {img_path}: {e}")

    if features:
        centroid = np.mean(features, axis=0)
        centroids[class_name] = centroid
        print(f"✅ Selesai: {len(features)} fitur diproses.")
    else:
        print("⚠️ Tidak ada fitur valid untuk kelas ini.")

# Simpan hasil centroid ke file
with open(CENTROID_SAVE_PATH, 'wb') as f:
    pickle.dump({'centroids': centroids, 'class_names': class_names}, f)

print(f"\n🎉 Semua centroid berhasil disimpan ke '{CENTROID_SAVE_PATH}'")
