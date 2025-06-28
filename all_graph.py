import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ===== Konfigurasi Dataset dan Model =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "dataset/v3"
MODEL_PATH = "models/v5.4.1/model_kain_tenun_mobilenetv2_crop.h5"

# ===== Center Crop + Normalisasi =====
def preprocess(image, label):
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# ===== Load Data Validasi =====
val_ds_raw = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = val_ds_raw.class_names

val_ds = val_ds_raw.map(preprocess)

# ===== Load Model =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Prediksi Data Validasi =====
y_true = []
y_pred = []
y_scores = []

for images, labels in val_ds:
    probs = model.predict(images)
    preds = np.argmax(probs, axis=1)
    actuals = np.argmax(labels.numpy(), axis=1)
    
    y_true.extend(actuals)
    y_pred.extend(preds)
    y_scores.extend(np.max(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

# ===== Confusion Matrix =====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ===== Classification Report =====
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# ===== Precision, Recall, F1 vs Confidence =====
bins = np.linspace(0, 1, 11)  # 10 bins: [0.0-0.1], ..., [0.9-1.0]
bin_indices = np.digitize(y_scores, bins) - 1

precision_per_bin = []
recall_per_bin = []
f1_per_bin = []
bin_centers = (bins[:-1] + bins[1:]) / 2

for i in range(len(bin_centers)):
    idxs = bin_indices == i
    if np.sum(idxs) == 0:
        precision_per_bin.append(np.nan)
        recall_per_bin.append(np.nan)
        f1_per_bin.append(np.nan)
        continue

    p, r, f1, _ = precision_recall_fscore_support(
        y_true[idxs], y_pred[idxs], average='macro', zero_division=0
    )
    precision_per_bin.append(p)
    recall_per_bin.append(r)
    f1_per_bin.append(f1)

# ===== Plotting Curve =====
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, precision_per_bin, label='Precision', marker='o')
plt.plot(bin_centers, recall_per_bin, label='Recall', marker='s')
plt.plot(bin_centers, f1_per_bin, label='F1 Score', marker='^')
plt.title("Precision / Recall / F1 vs Confidence")
plt.xlabel("Confidence")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.savefig("confussionMatrix.png")
plt.show()
