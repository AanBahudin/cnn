import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

# ========= KONFIGURASI =========
IMG_SIZE = (224, 224)
RESIZE_SIZE = (256, 256)  # Resize ke ukuran lebih besar dulu untuk di-crop
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "dataset/v3"

# ========= Fungsi Resize + Center Crop =========
def resize_and_center_crop(image, label):
    # Resize gambar terlebih dahulu
    image = tf.image.resize(image, RESIZE_SIZE)

    # Crop tengah (center crop ke 224x224)
    crop_height, crop_width = IMG_SIZE
    offset_height = (RESIZE_SIZE[0] - crop_height) // 2
    offset_width = (RESIZE_SIZE[1] - crop_width) // 2
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)

    # Normalisasi
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# ========= Load Dataset =========
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=RESIZE_SIZE,  # ukurannya harus sama seperti di resize
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=RESIZE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# ========= Resize + Center Crop + Normalisasi =========
train_ds = train_ds.map(resize_and_center_crop).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(resize_and_center_crop).prefetch(tf.data.AUTOTUNE)

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

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========= Training =========
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ========= Simpan Model =========
model.save("model_kain_tenun.h5")
print("✅ Model berhasil disimpan sebagai 'model_kain_tenun.h5'")
