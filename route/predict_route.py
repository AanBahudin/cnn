from flask import Blueprint, request, jsonify
from controller.predict_controller import classify_image, preprocess_image
from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

# Load model sekali saja di awal
model = tf.keras.models.load_model("models/v5.4/model_kain_tenun_mobilenetv2_crop.h5")

# Label kelas sesuai urutan folder saat training
class_names = [
    'Bancana Kaluku Bula Laki laki', 
    'Bancana Kaluku Bula Perempuan', 
    'Katamba Layana Laki laki', 
    'Katamba Layana Perempuan',
    'Manggopa Laki laki',
    'Manggopa Perempuan',
    'Samasili Laki laki',
    'Samasili Perempuan'
]

predict_blueprint = Blueprint('predict', __name__)


# Fungsi resize + padding
def resize_with_padding(img, target_size=(224, 224), color=(0, 0, 0)):
    img.thumbnail(target_size, Image.Resampling.LANCZOS)  # untuk Pillow 10+
    delta_w = target_size[0] - img.size[0]
    delta_h = target_size[1] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=color)

@predict_blueprint.route('/data/processing', methods=['POST'])
def process_post():
    if 'motifKain' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['motifKain']
    
    try:
        # Baca gambar dari request dan ubah ke RGB
        img = Image.open(image_file.stream).convert("RGB")
        
        # Resize + padding agar tidak distorsi
        img = resize_with_padding(img, (224, 224))

        # Preprocess
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        return jsonify({
            'label': predicted_label,
            'confidence': f"{confidence:.2%}",
            'confidence_raw': confidence
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
