import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# === GANTI SESUAI KONDISI KAMU ===
model_path = 'models/v7.2/model_cnn.h5'
test_dir = 'dataset/TRAINING'
img_size = (224, 224)  # Sesuaikan dengan input model kamu
batch_size = 32

# === LOAD MODEL ===
model = tf.keras.models.load_model(model_path)

# === LOAD DATASET TEST DARI FOLDER ===
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Ambil nama kelas
class_names = test_dataset.class_names

# Normalisasi piksel
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# === PREDIKSI DAN KUMPULKAN NILAI ===
y_true, y_pred, y_conf = [], [], []

for images, labels in test_dataset:
    preds = model.predict(images)
    pred_labels = np.argmax(preds, axis=1)
    conf_scores = np.max(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)
    y_conf.extend(conf_scores)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_conf = np.array(y_conf)

# === HITUNG METRIK PER KELAS ===
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=range(len(class_names)), zero_division=0
)

# === RATA-RATA CONFIDENCE PER KELAS YANG DIPREDIKSI ===
avg_conf_per_class = []
for i in range(len(class_names)):
    class_indices = np.where(y_pred == i)[0]
    if len(class_indices) > 0:
        avg_conf = np.mean(y_conf[class_indices])
    else:
        avg_conf = 0
    avg_conf_per_class.append(avg_conf)

# === FUNGSI UNTUK BAR CHART CUSTOM ===
def plot_metric_vs_confidence(conf, metric, ylabel, title):
    x = np.arange(len(conf))  # gunakan indeks sebagai x-axis
    plt.figure(figsize=(10, 4))
    bars = plt.bar(x, metric, width=0.5, color='skyblue', edgecolor='black')

    # Label nilai metrik di atas batang
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

    plt.xticks(x, [f"{c:.2f}" for c in conf])
    plt.xlabel('Confidence')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

# === TAMPILKAN SEMUA GRAFIK ===
plot_metric_vs_confidence(avg_conf_per_class, f1, 'F1 Score', 'F1 Score vs Confidence')
plot_metric_vs_confidence(avg_conf_per_class, precision, 'Precision', 'Precision vs Confidence')
plot_metric_vs_confidence(avg_conf_per_class, recall, 'Recall', 'Recall vs Confidence')

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()