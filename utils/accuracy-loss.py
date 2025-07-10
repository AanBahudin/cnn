import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ===== PATH ke model dan dataset test =====
MODEL_PATH = "models/v7.2/model_cnn.h5"
TEST_DIR = "dataset/TESTING"

# ===== Load dataset test =====
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Simpan label kelas dari folder
class_names = raw_test_ds.class_names
print("Label Kelas:", class_names)

# ===== (Jika Perlu) Preprocessing untuk MobileNetV2 =====
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess(images, labels):
    labels = tf.one_hot(labels, depth=len(class_names))  # ubah label ke one-hot
    return preprocess_input(images), labels

test_ds = raw_test_ds.map(preprocess)

# ===== Load model =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Evaluasi langsung =====
loss, accuracy = model.evaluate(test_ds)
print(f"\n📉 Loss: {loss:.4f}")
print(f"✅ Accuracy: {accuracy:.2%}")

# ===== Prediksi manual =====
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predicted_labels = tf.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels.numpy())

# Konversi y_true dari one-hot ke label indeks
y_true = np.argmax(y_true, axis=1)  # <<-- Tambahkan baris ini

# ===== Classification Report =====
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===== Confusion Matrix =====
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== Buat Kurva Dummy dari Satu Titik Akurasi dan Loss =====
# Kita hanya punya satu titik (test loss & accuracy), jadi buat visualisasi sederhana

plt.figure(figsize=(8, 5))

# Kurva Accuracy (hanya 1 titik)
plt.subplot(1, 2, 1)
plt.plot([1], [accuracy], 'bo-', label='Test Accuracy')
plt.title("Akurasi Model")
plt.ylim(0, 1)
plt.xticks([1], ["Test Set"])
plt.ylabel("Akurasi")
plt.legend()

# Kurva Loss (hanya 1 titik)
plt.subplot(1, 2, 2)
plt.plot([1], [loss], 'ro-', label='Test Loss')
plt.title("Loss Model")
plt.ylim(0, max(1.0, loss + 0.1))
plt.xticks([1], ["Test Set"])
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
