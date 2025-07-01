# 🌿 Chilitify: Deteksi Penyakit Daun Cabai Menggunakan DenseNet121

## 📘 Deskripsi Proyek

**Chilitify** adalah sistem klasifikasi citra berbasis deep learning yang dirancang untuk mendeteksi berbagai jenis **penyakit pada daun cabai**. Proyek ini menggunakan arsitektur **DenseNet121** yang telah terbukti efisien dan akurat untuk tugas klasifikasi gambar.

Sistem ini dilatih pada dataset gambar daun cabai yang memiliki lima kelas:
1. **Daun Sehat**
2. **Daun Terkena Kuning**
3. **Daun Terkena Keriting (Leaf Curl)**
4. **Daun Terkena Bercak Daun (Leaf Spot)**
5. **Daun Terkena Hama Whitefly**

---

## 🎯 Tujuan Proyek

- Membangun model klasifikasi gambar untuk mendeteksi penyakit daun cabai.
- Meningkatkan akurasi deteksi penyakit tanaman secara otomatis.
- Memberikan kontribusi pada sistem pertanian cerdas berbasis teknologi.

---

## 🧰 Tools & Teknologi

- Python 3
- TensorFlow & Keras
- DenseNet121 (Transfer Learning)
- Matplotlib & Seaborn
- Google Colab
- Google Cloud Platform (untuk deployment opsional)
- h5py (untuk menyimpan model)

---

## 🖼️ Dataset

Dataset berisi gambar daun cabai dengan lima label kategori:

| Label          | Keterangan                |
|----------------|---------------------------|
| `healthy`      | Daun sehat (tidak terinfeksi) |
| `yellowish`    | Gejala menguning           |
| `leaf_curl`    | Daun melengkung atau keriting |
| `leaf_spot`    | Daun bercak                |
| `whitefly`     | Infeksi hama kutu putih    |

> 📌 Dataset diperoleh dari Kaggle (penyebutan sumber dalam notebook). Setiap gambar berukuran 224x224 piksel.

---

## ⚙️ Arsitektur Model

Model menggunakan **Transfer Learning** dengan arsitektur **DenseNet121**. Adaptasi dilakukan dengan menambahkan layer fully connected di bagian atas (top layers).

### Struktur:
- Base Model: `DenseNet121` (tanpa top layer, pretrained di ImageNet)
- GlobalAveragePooling2D
- Dropout
- Dense (5 kelas, aktivasi softmax)

---

## 🔄 Alur Proses

1. **Import Dataset** dan Visualisasi Awal
2. **Preprocessing Gambar**: Resize, augmentasi, normalisasi
3. **Split Data**: Train (80%) dan Validation (20%)
4. **Load Model DenseNet121**
5. **Training Model**
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Metrics: Accuracy
6. **Evaluasi Model**
   - Plot akurasi dan loss
   - Confusion Matrix
   - Classification Report
7. **Export Model** ke format `.h5`
8. **(Opsional)** Deployment menggunakan GCP Cloud Run

---

## ✅ Hasil Evaluasi

Model mencapai akurasi validasi yang sangat baik:

- **Akurasi Validasi:** 97%
- **Overfitting:** Tidak signifikan (akurasi training dan validasi seimbang)
- **Confusion Matrix:** menunjukkan klasifikasi antar kelas yang tepat

---

## 📦 Simpan dan Load Model
```
# Simpan model
model.save("chilitify_densenet_model.h5")

# Load model
from tensorflow.keras.models import load_model
model = load_model("chilitify_densenet_model.h5")
'''

## 🧪 Contoh Prediksi Gambar Baru
'''
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = "sample_leaf.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print("Prediksi:", class_names[predicted_class])
'''

