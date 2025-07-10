# Program CNN Klasifikasi Sarung (Enhanced + Smart Crop)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image

# ==== Konfigurasi ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 110
RAW_DATASET_DIR = "dataset/v4"
CROPPED_DATASET_DIR = "dataset_cropped/v4"

# ==== Smart Crop Utility ====
def smart_crop(image_path, save_path):
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        resized = cv2.resize(img, IMG_SIZE)
        cv2.imwrite(save_path, resized)
        return

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area_ratio = (w * h) / (w_img * h_img)

    if area_ratio > 0.9:
        cropped = img
    else:
        cropped = img[y:y+h, x:x+w]

    resized = cv2.resize(cropped, IMG_SIZE)
    cv2.imwrite(save_path, resized)

# ==== Terapkan Smart Crop ke Semua Dataset ====
def prepare_cropped_dataset():
    if not os.path.exists(CROPPED_DATASET_DIR):
        os.makedirs(CROPPED_DATASET_DIR)

    for class_name in os.listdir(RAW_DATASET_DIR):
        src_folder = os.path.join(RAW_DATASET_DIR, class_name)
        dst_folder = os.path.join(CROPPED_DATASET_DIR, class_name)
        os.makedirs(dst_folder, exist_ok=True)

        for img_name in os.listdir(src_folder):
            src_path = os.path.join(src_folder, img_name)
            dst_path = os.path.join(dst_folder, img_name)
            smart_crop(src_path, dst_path)

prepare_cropped_dataset()

# ==== Augmentasi Data ====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# ==== Normalisasi ====
def normalize(image, label):
    normalized = tf.cast(image, tf.float32) / 255.0
    return normalized, label

# ==== Dataset Training dan Validasi ====
train_ds = image_dataset_from_directory(
    CROPPED_DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = image_dataset_from_directory(
    CROPPED_DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ==== Load Pretrained MobileNetV2 ====
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# ==== Bangun Model ====
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==== Callback EarlyStopping ====
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ==== Training Model ====
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ==== Simpan Model ====
model.save("model_cnn.h5")
print("\u2705 Model berhasil disimpan sebagai 'model_cnn.h5'")

# ==== Visualisasi Hasil Training ====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()