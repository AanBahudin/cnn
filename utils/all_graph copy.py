import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ===== Konfigurasi Dataset dan Model =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "dataset/v4"
MODEL_PATH = "models/v6.1/model_kain_tenun_mobilenetv2_crop.h5"

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
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    actuals = np.argmax(labels.numpy(), axis=1)
    
    y_true.extend(actuals)
    y_pred.extend(preds)
    y_scores.extend(np.max(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

# ===== Confusion Matrix (diperbaiki tampilannya) =====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, annot_kws={"size": 7})
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_small.png")
plt.show()

# ===== Classification Report (Text) =====
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# ===== Binning berdasarkan confidence =====
bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(y_scores, bins) - 1
bin_centers = (bins[:-1] + bins[1:]) / 2

# ===== Kumpulkan skor untuk tiap bin =====
precision_per_bin, recall_per_bin, f1_per_bin = [], [], []

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

# ===== Grafik Waterfall (bar chart) =====
def plot_bar_chart(data, title, ylabel, filename):
    plt.figure(figsize=(8, 4))
    bars = plt.bar(bin_centers, data, width=0.08, color='skyblue', edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=7)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel(ylabel)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_bar_chart(precision_per_bin, "Precision vs Confidence", "Precision", "precision_vs_confidence.png")
plot_bar_chart(recall_per_bin, "Recall vs Confidence", "Recall", "recall_vs_confidence.png")
plot_bar_chart(f1_per_bin, "F1 Score vs Confidence", "F1 Score", "f1_vs_confidence.png")
