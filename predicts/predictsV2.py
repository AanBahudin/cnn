import tensorflow as tf
from tensorflow.keras import layers, models

img_size = (224, 224)
batch_size = 32

# 1. Load dataset dengan split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/v2",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/v2",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# 2. Normalisasi piksel
normalization_layer = layers.Rescaling(1./255)

# 3. Augmentasi untuk data training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(factor=0.2),
    layers.RandomTranslation(0.1, 0.1)
])

# 4. Terapkan augmentasi hanya pada train_ds
train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x), training=True), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 5. Prefetch agar lebih cepat
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 6. Buat dan latih model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_ds.element_spec[1].shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

# 7. Simpan model
model.save("model_kain_tenun.h5")
print("Model berhasil disimpan sebagai 'model_kain_tenun.h5'")
