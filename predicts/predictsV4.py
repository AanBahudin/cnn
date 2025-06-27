import os
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split

# ======== KONFIGURASI =========
IMG_SIZE = (224, 224)
DATASET_DIR = "dataset/v2"
BATCH_SIZE = 32

# ======== Fungsi Resize + Padding =========
def resize_with_padding(img, target_size=(224, 224), color=(0, 0, 0)):
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding, fill=color)

# ======== Load Dataset Manual=========
X = []
y = []
class_names = sorted(os.listdir(DATASET_DIR))

for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"Lewati file tak didukung: {fname}")
            continue
        
        fpath = os.path.join(class_dir, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            img = resize_with_padding(img, IMG_SIZE)
            img_array = np.array(img) / 255.0  # Normalisasi
            X.append(img_array)
            y.append(label_idx)
        except Exception as e:
            print(f"Gagal proses gambar {fpath}: {e}")

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))

# ======== Split Train / Val =========
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ======== Konversi ke tf.data.Dataset =========
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ======== Model CNN =========
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

# ======== Simpan Model =========
model.save("model_kain_tenun.h5")
print("Model berhasil disimpan sebagai 'model_kain_tenun.h5'")
