import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load model
model = tf.keras.models.load_model("models/v2/model_kain_tenun.h5")

# 2. Path ke gambar uji
img_path = "test/Samasili Test/sarung (3).jpg"

# 3. Parameter sesuai training
img_size = (224, 224)

# 4. Baca dan preprocess gambar
img = Image.open(img_path).convert("RGB")
img = img.resize(img_size)
img_array = np.array(img) / 255.0  # normalisasi
img_array = np.expand_dims(img_array, axis=0)  # bentuknya jadi (1, 224, 224, 3)

# 5. Prediksi
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])

# 6. Label kelas (HARUS SESUAI URUTAN folder train saat training)
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

# 7. Cetak hasil
print(f"Hasil prediksi: {class_names[predicted_index]}")
print(f"Probabilitas: {predictions[0][predicted_index]:.2%}")