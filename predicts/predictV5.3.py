import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ==== Konfigurasi ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_DIR = "dataset/v3"

# ==== Fungsi Center Crop + Normalisasi ====
def center_crop_and_resize(image, label):
    cropped = tf.image.central_crop(image, central_fraction=0.875)  # potong tengah (87.5% dari tinggi/lebar)
    resized = tf.image.resize(cropped, IMG_SIZE)
    normalized = tf.cast(resized, tf.float32) / 255.0
    return normalized, label

# ==== Dataset Training dan Validasi ====
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),  # pakai ukuran besar agar crop punya ruang
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

# ==== Terapkan center crop dan normalisasi ====
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(center_crop_and_resize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(center_crop_and_resize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ==== Load Pretrained MobileNetV2 ====
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# ==== Build Model ====
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

# ==== Training ====
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ==== Simpan Model ====
model.save("model_kain_tenun_mobilenetv2_crop.h5")
print("✅ Model berhasil disimpan sebagai 'model_kain_tenun_mobilenetv2_crop.h5'")
