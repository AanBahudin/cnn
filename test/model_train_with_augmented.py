# Program CNN Klasifikasi Sarung (Enhanced Version)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ==== Konfigurasi ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 110
DATASET_DIR = "dataset/v4"

# ==== Augmentasi Data ====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# ==== Fungsi Crop dan Normalisasi ====
def center_crop_and_resize(image, label):
    cropped = tf.image.central_crop(image, central_fraction=0.875)
    resized = tf.image.resize(cropped, IMG_SIZE)
    normalized = tf.cast(resized, tf.float32) / 255.0
    return normalized, label

# ==== Dataset Training dan Validasi ====
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

# ==== Terapkan Augmentasi, Crop, dan Normalisasi ====
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(center_crop_and_resize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(center_crop_and_resize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

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

# ==== [Optional] Fine-Tuning ====
# base_model.trainable = True
# for layer in base_model.layers[:100]:
#     layer.trainable = False
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
#               loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stop])
# model.save("model_cnn_finetuned.h5")
