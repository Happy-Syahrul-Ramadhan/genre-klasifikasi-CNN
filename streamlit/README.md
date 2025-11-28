# Music Genre Classification - Streamlit App

Aplikasi web interaktif untuk klasifikasi genre musik menggunakan Convolutional Neural Network (CNN).

## 🎵 Fitur

- **Upload File MP3**: Upload file musik dalam format MP3
- **Prediksi Genre**: Model CNN memprediksi genre (Ambient, Pop, Rock)
- **Visualisasi Audio**:
  - Waveform (gelombang audio)
  - Spectrogram (representasi frekuensi-waktu)
  - Mel-Spectrogram (skala mel untuk persepsi manusia)
  - MFCCs (fitur yang digunakan untuk prediksi)
- **Confidence Score**: Tingkat keyakinan prediksi model
- **Multi-segment Analysis**: Analisis 10 segmen audio (3 detik per segmen)

## 🚀 Cara Menjalankan Lokal

### 1. Install Dependencies

```bash
cd streamlit
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## ☁️ Deploy ke Streamlit Cloud

### Prerequisites
1. Model file `model_cnn2.h5` harus ada di folder `models/`
2. Repository di-push ke GitHub

### Langkah Deploy:

1. **Push ke GitHub**
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

2. **Streamlit Cloud Setup**
   - Kunjungi [share.streamlit.io](https://share.streamlit.io)
   - Login dengan GitHub
   - Click "New app"
   - Pilih repository: `Happy-Syahrul-Ramadhan/genre_music_classification_with_CNN`
   - Main file path: `streamlit/app.py`
   - Click "Deploy"

3. **Troubleshooting Deployment**
   - Jika error "dependencies", pastikan `requirements.txt` dan `packages.txt` ada
   - Jika error "model not found", pastikan path model relatif (bukan absolute)
   - Jika error "memory", model `.h5` mungkin terlalu besar (gunakan Git LFS)

## 📁 Struktur Folder

```
streamlit/
├── app.py              # Aplikasi utama Streamlit
├── requirements.txt    # Python dependencies
└── README.md          # Dokumentasi ini
```

## 🎯 Cara Penggunaan

1. **Upload File**
   - Klik tombol "Browse files" di sidebar
   - Pilih file MP3 yang ingin diklasifikasi
   - File akan diproses secara otomatis

2. **Lihat Hasil**
   - Genre yang diprediksi ditampilkan dengan confidence score
   - Bar chart menampilkan probabilitas untuk semua genre
   - Eksplorasi visualisasi audio di berbagai tab

3. **Analisis Detail**
   - Buka "View Detailed Results" untuk melihat:
     - Prediksi per segmen
     - Informasi teknis (shape MFCC, durasi, dll)

## 🧠 Model Information

- **Arsitektur**: CNN dengan Regularization & Data Augmentation
- **Input**: 13 MFCCs per segmen audio 3 detik
- **Output**: 3 genre (Ambient, Pop, Rock)
- **Training Data**: ~910 segmen audio
- **Model Path**: `../models/model_cnn3.h5`

## ⚙️ Parameter Audio Processing

- **Sampling Rate**: 22,050 Hz
- **N_MFCC**: 13
- **N_FFT**: 2048
- **Hop Length**: 512
- **Duration**: 30 detik (maksimal)
- **Segments**: 10 segmen @ 3 detik

## 📊 Visualisasi

### 1. Waveform
Representasi amplitudo audio dalam domain waktu

### 2. Spectrogram
Representasi frekuensi audio dari waktu ke waktu menggunakan STFT

### 3. Mel-Spectrogram
Spectrogram dengan skala frekuensi mel (mirip persepsi pendengaran manusia)

### 4. MFCCs
Koefisien cepstral frekuensi mel yang digunakan sebagai input model

## 🔧 Troubleshooting

### Error: Model not found
```
Error loading model: [Errno 2] No such file or directory
```
**Solusi**: Pastikan file `model_cnn3.h5` ada di folder `models/`

### Error: Memory error
**Solusi**: File audio terlalu besar. Coba file dengan durasi lebih pendek (<30 detik)

### Error: Invalid audio file
**Solusi**: Pastikan file dalam format MP3 yang valid

## 📝 Catatan

- Model dilatih pada 3 genre: Ambient, Pop, Rock
- File audio lebih panjang dari 30 detik akan dipotong
- Hasil terbaik dengan audio berkualitas baik
- Prediksi diambil dari rata-rata 10 segmen

## 🎨 Kustomisasi

Untuk mengubah genre atau model:

1. Edit `GENRES` list di `app.py`
2. Ubah `MODEL_PATH` ke model yang diinginkan
3. Sesuaikan parameter audio jika perlu

## 📄 License

Project ini adalah bagian dari tugas Music Genre Classification menggunakan CNN.
