from PIL import Image
import numpy as np
import io
import base64

def classify_image(image_file):
    return 'test'

def preprocess_image(image_bytes):
    # Buka dan konversi gambar ke grayscale
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((224, 224))

    # Simpan hasil grayscale untuk ditampilkan kembali dalam base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    base64_result = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Buat array untuk CNN (1 channel, karena grayscale)
    img_array = np.array(img) / 255.0  # shape: (224, 224)
    img_array = np.expand_dims(img_array, axis=0)    # shape: (1, 224, 224)
    img_array = np.expand_dims(img_array, axis=-1)   # shape: (1, 224, 224, 1)

    return base64_result, img_array