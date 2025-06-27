import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Ukuran gambar
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load dataset dari folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Normalisasi piksel 0-1
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Bangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # karena 2 kelas
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Latih model
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Evaluasi akhir
loss, acc = model.evaluate(val_ds)
print(f"Akurasi: {acc:.2f}")
 