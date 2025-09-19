# Sistem Deteksi Objek Person dengan YOLOv8 & OpenCV

Sistem deteksi objek yang dibangun menggunakan YOLOv8 dan OpenCV, mampu mendeteksi dan melacak berbagai objek Person dalam gambar, video dan kamera dengan akurasi tinggi dan performa real-time.

## 🛠️ Instalasi

### Clone repository:

```bash
git clone https://github.com/jutionck/object-detection-with-yolo-opencv.git
cd object-detection-with-yolo-opencv
```

### Buat virtual environment (opsional tapi direkomendasikan):

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### Install dependensi:

```bash
pip install -r requirements.txt
```

📦 Dependensi

- Python 3.8+
- OpenCV (4.10.0.84)
- Ultralytics YOLOv8 (8.2.0)
- NumPy (1.26.4)
- PyTorch (>=2.0.0)
- TorchVision (>=0.15.0)
- Matplotlib (>=3.5.0)

### Install manual jika diperlukan:

```bash
pip install opencv-python ultralytics numpy torch torchvision matplotlib
```

## ▶️ Penggunaan

**Jalankan dengan video default:**

```bash
python main.py
```

**Jalankan dengan webcam:**

```bash
python main.py --webcam
# atau
python main.py -w
```

**Jalankan dengan file video lain:**

```bash
python main.py path/ke/video.mp4
```

Sistem akan menampilkan deteksi person real-time dengan statistik lengkap. Tekan 'q' untuk keluar dan menghasilkan laporan akhir.

## 📋 **Output Laporan:**

- **Real-time Display**: Statistik langsung di layar video
- **Console Log**: Laporan berkala setiap 30 frame
- **JSON Report**: Data lengkap dengan timestamp dan koordinat
- **CSV File**: Data tabular untuk analisis lebih lanjut
- **Grafik PNG**: Visualisasi timeline dan distribusi deteksi
