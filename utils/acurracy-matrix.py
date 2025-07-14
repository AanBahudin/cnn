import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ===== Path ke model dan dataset =====
MODEL_PATH = "models/v7.3/model_cnn.h5"
# MODEL_PATH = "model_cnn.h5"
TEST_DIR = "dataset/TESTING"

# ===== Load dataset test =====
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),       # sesuaikan dengan input model kamu
    batch_size=32,
    shuffle=False
)

# Simpan label kelas dari struktur folder
class_names = raw_test_ds.class_names
print("Label Kelas:", class_names)

# ===== (Opsional) Preprocessing untuk MobileNetV2 =====
# Hanya gunakan ini jika saat training kamu pakai MobileNetV2 tanpa Rescaling() layer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess(images, labels):
    return preprocess_input(images), labels

# Terapkan preprocessing
test_ds = raw_test_ds.map(preprocess)

# ===== Load model =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Evaluasi manual =====
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predicted_labels = tf.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels.numpy())

# Hitung akurasi
correct = sum([1 for a, b in zip(y_true, y_pred) if a == b])
accuracy = correct / len(y_true)
print(f"\n✅ Akurasi Model: {accuracy:.2%}")

# ===== Tampilkan classification report =====
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===== Tampilkan confusion matrix =====
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
