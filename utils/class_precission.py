import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd

# ====== Ganti dengan hasil prediksi kamu ======
# Misalnya y_true dan y_pred sudah dihitung di program sebelumnya
# dan class_names sudah kamu ambil dari image_dataset_from_directory

# Contoh dummy (hapus jika sudah punya y_true, y_pred, class_names)
# y_true = [0, 1, 2, 1, 0, 2, 1]
# y_pred = [0, 2, 2, 1, 0, 2, 1]
# class_names = ['Kelas A', 'Kelas B', 'Kelas C']

# ====== Buat classification report dalam bentuk dict ======
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    digits=4
)

# ====== Konversi ke DataFrame dan ambil hanya precision ======
df = pd.DataFrame(report).transpose()
df_precision = df.loc[class_names, 'precision']

# ====== Plot Precision Per Kelas dalam Curve Chart ======
plt.figure(figsize=(10, 6))
plt.plot(class_names, df_precision, marker='o', linestyle='-', color='blue', linewidth=2)
plt.xticks(rotation=45, ha='right')
plt.title("Precision per Kelas (Curve Chart)")
plt.xlabel("Kelas")
plt.ylabel("Precision")
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
