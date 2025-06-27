import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import ImageOps
import numpy as np

# ========= KONFIGURASI =========
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
DATASET_DIR = "dataset/v3"

# ========= Fungsi Resize + Padding =========
def resize_with_padding_tf(image, label):
    # Dapatkan ukuran asli gambar
    shape = tf.shape(image)[:2]
    h, w = shape[0], shape[1]

    # Rasio perbandingan
    ratio = tf.minimum(IMG_SIZE[0] / tf.cast(w, tf.float32), IMG_SIZE[1] / tf.cast(h, tf.float32))
    new_w = tf.cast(ratio * tf.cast(w, tf.float32), tf.int32)
    new_h = tf.cast(ratio * tf.cast(h, tf.float32), tf.int32)

    # Resize dan padding
    image = tf.image.resize(image, (new_h, new_w))
    image = tf.image.resize_with_pad(image, IMG_SIZE[0], IMG_SIZE[1])
    image = tf.cast(image, tf.float32) / 255.0  # normalisasi
    return image, label

# ========= Load Dataset =========
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),  # sementara lebih besar, nanti diproses ulang
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

# ========= Resize + Padding + Normalisasi =========
train_ds = train_ds.map(resize_with_padding_tf).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(resize_with_padding_tf).prefetch(tf.data.AUTOTUNE)

# ========= Model =========
class_names = [
    'Bancana Kaluku Laki laki', 
    'Bancana Kaluku Perempuan', 
    'Katamba Layana Laki laki', 
    'Katamba Layana Perempuan',
    'Manggopa Laki laki',
    'Manggopa Perempuan',
    'Samasili Laki laki',
    'Samasili Perempuan'
]
num_classes = len(class_names)

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# ========= Training =========
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ========= Simpan Model =========
model.save("model_kain_tenun.h5")
print("✅ Model berhasil disimpan sebagai 'model_kain_tenun.h5'")