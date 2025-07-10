import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ==== Konfigurasi ====
IMG_SIZE = (224, 224)       #   ukuran gambar yang digunakan untuk training
BATCH_SIZE = 32             #   jumlah gambar yang akan dipreprocessing dalam 1 waktu
EPOCHS =     17            #   jumlah pendekatan
TEST_DATASET = "dataset/TESTING"  #   letak dataset train yang digunakan untuk pembuatan model
TRAINING_DATASET = "dataset/TRAINING"  #   letak dataset test yang digunakan untuk pembuatan model

# ==== Fungsi Center Crop + Normalisasi ====
def center_crop_and_resize(image, label):   
    cropped = tf.image.central_crop(image, central_fraction=0.875)
    resized = tf.image.resize(cropped, IMG_SIZE)
    normalized = tf.cast(resized, tf.float32) / 255.0
    return normalized, label

# ==== Dataset Training dan Validasi ====
train_ds = image_dataset_from_directory(
    TRAINING_DATASET,                #   letak dataset
    image_size=(256, 256),      #   pakai ukuran besar agar crop punya ruang
    batch_size=BATCH_SIZE,      #   jumlah batch yang akan digunakan dalam 1 kali proses preprocessing
    label_mode="categorical"    #   jenis data yang digunakan
)

val_ds = image_dataset_from_directory(  # sama seperti penjelasan diatas
    TEST_DATASET,
    image_size=(256, 256),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names  #   mengambil nama dari kelas yang ada. contoh Bancana Kaluku Bula Laki laki sampai dengan Samasili Perempuan
num_classes = len(class_names)      #   mengambil jumlah kelas yang ada. yang kita punya total 8

# ==== Terapkan center crop dan normalisasi ====
AUTOTUNE = tf.data.AUTOTUNE         #   AUTOTONE digunakan untuk menghitung jumlah thread ideal untuk pemrosesan data agar GPU atau CPU tidak overload
train_ds = train_ds.map(center_crop_and_resize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)     #   data mulai dilatih, prefetch maksudnya adalah untuk menyiapkan batch gambar berikutnya. contoh gambar 1 - 32 sedang di preprocessing maka prefetch akan menyiapkan gambar 33 - 64 untuk di preprocessing segera setelah gambar 1 - 32 selesai.
val_ds = val_ds.map(center_crop_and_resize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)     #   sama dengan penjelasan diatas

# ==== Load Pretrained MobileNetV2 ====
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),   # load gambar dengan konfigurasi 224, 224, 3. IMG_SIZE = 224 * 224. 3 maksudnya adalah gambar 3 channel (berwarna)
    include_top=False,      #   kita tidak menggunakan layer akhir bawaan dari MobileNetV2. tetapi kita punya sendiri yaitu jumlah dataset
    weights='imagenet'      #   menggunakan fitur imagenet yang sudah disediakan oleh MobileNetV2 untuk belajar fitur garis, tepi, bentuk
)
base_model.trainable = False  # Freeze base model

# ==== Build Model ====
model = models.Sequential([
    base_model,                             #  mengambil base model dari MobileNetV2
    layers.GlobalAveragePooling2D(),        #  mengubah / meratakan gambar 2D menjadi 1 D  
    layers.Dense(128, activation='relu'),   #  membuat 128 neuron dengan masing masing neuron menerima fitur dari layer diatas untuk menghitung outputnya sendiri   
    # layers.Dropout(0.3),                    #  Dropout adalah teknik “mematikan” (mengabaikan) sejumlah neuron secara acak saat training
    layers.Dense(num_classes, activation='softmax')     #   meng-generate nilai nilai probabilitas untuk tiap tiap kelas. nilai tertinggi dari semua kelas akan menjadi hasil klasifikasi
])

model.compile(
    optimizer='adam',                   #   model di optimisasi menggunakan metode adam
    loss='categorical_crossentropy',    #   perhitungan loss atau kesalahan pada saat training. categorical_crossentropy digunakan karena data yang digunakan merupakan jenis categorical
    metrics=['accuracy']                #   yang di ukur pada proses training adalah akurasi
)

# ==== Training ====
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)  #   model mulai dilatih. fungsi fit membutuhkan argumen train_ds (dataset yang digunakan untuk training), validation_data = val_ds (dataset yang digunakan untuk validasi atau uji) dan epoch (jumlah pendekatan yang akan digunakan ketika data di latih)

# ==== Simpan Model ====
model.save("model_cnn.h5")  # data yang telah dilatih disimpan dengan nama model_kain_tenun_mobilenetv2_crop.h5
print("✅ Model berhasil disimpan sebagai 'model_cnn.h5'")  #   cetak pesan jika data sudah selesai di training
