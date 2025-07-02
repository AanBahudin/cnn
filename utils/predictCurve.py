import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# ==== Konfigurasi ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "dataset/v4"
MODEL_PATH = "models/v6.1/model_kain_tenun_mobilenetv2_crop.h5"

# ==== Fungsi center crop dan normalisasi ====
def center_crop_and_resize(image, label):
    cropped = tf.image.central_crop(image, central_fraction=0.875)
    resized = tf.image.resize(cropped, IMG_SIZE)
    normalized = tf.cast(resized, tf.float32) / 255.0
    return normalized, label

# ==== Ambil val_ds dan class_names ====
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)
class_names = val_ds_raw.class_names
val_ds = val_ds_raw.map(center_crop_and_resize)

# ==== Load Model ====
model = tf.keras.models.load_model(MODEL_PATH)

# ==== Prediksi ====
y_true = []
y_pred = []
y_confidence = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())
    y_confidence.extend(tf.reduce_max(preds, axis=1).numpy())  # confidence tertinggi per prediksi

# ==== Metrics ====
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
precision = [report[cls]['precision'] for cls in class_names]
recall = [report[cls]['recall'] for cls in class_names]

# ==== Confidence hanya untuk prediksi yang benar ====
conf_per_class = []
for idx in range(len(class_names)):
    confs = [y_confidence[i] for i in range(len(y_pred)) if y_pred[i] == idx and y_pred[i] == y_true[i]]
    conf_per_class.append(np.mean(confs) if confs else 0)

# ==== Grafik Line: Precision ====
plt.figure(figsize=(10, 6))
plt.plot(class_names, precision, marker='o', linestyle='-', color='royalblue', label='Precision')
plt.title("Precision per Kelas")
plt.ylabel("Precision")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==== Grafik Line: Recall
