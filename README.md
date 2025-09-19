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
pip install -r Requirements.txt
```

## 📦 Dependensi

### Core Dependencies:
- Python 3.8+
- OpenCV (4.10.0.84)
- Ultralytics YOLOv8 (8.2.0)
- NumPy (1.26.4)
- PyTorch (>=2.0.0)
- TorchVision (>=0.15.0)
- Matplotlib (>=3.5.0)

### Web Dashboard Dependencies:
- Flask (>=2.3.0)
- Flask-SocketIO (>=5.3.0)
- Python-SocketIO (>=5.8.0)

### Install manual jika diperlukan:

```bash
# Core dependencies
pip install opencv-python ultralytics numpy torch torchvision matplotlib

# Web dependencies
pip install flask flask-socketio python-socketio
```

## ▶️ Penggunaan

Sistem memiliki 2 mode operasi:

### 🖥️ **Mode Console (main.py)**

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

**Jalankan dengan IP Camera/NVR:**
```bash
# RTSP stream
python main.py rtsp://admin:password@192.168.1.100:554/stream

# HTTP stream
python main.py http://192.168.1.100:8080/video.cgi
```

Sistem akan menampilkan deteksi person real-time dengan statistik lengkap. Tekan 'q' untuk keluar dan menghasilkan laporan akhir.

### 🌐 **Mode Web Dashboard (app.py)**

**Jalankan web dashboard:**
```bash
# Recommended - menggunakan launcher
python run_web.py

# Direct - langsung Flask app
python app.py
```

**Akses dashboard:**
Buka browser: **http://localhost:9000**

**Fitur Web Dashboard:**
- 🎥 **Live Video Feed**: Stream video real-time dengan bounding boxes
- 📊 **Real-time Statistics**: FPS, total deteksi, rata-rata, dll
- 📈 **Interactive Charts**: Timeline & distribusi menggunakan Chart.js
- 🎮 **Interactive Controls**: Start/stop, pilih sumber input
- 📱 **Responsive Design**: Desktop & mobile friendly
- 🔌 **Multiple Input Sources**: 
  - Video default (`sample.mp4`)
  - Webcam
  - IP Camera/NVR (RTSP/HTTP)

## 📋 **Output Laporan:**

### Console Mode:
- **Real-time Display**: Statistik langsung di layar video
- **Console Log**: Laporan berkala setiap 30 frame
- **JSON Report**: Data lengkap dengan timestamp dan koordinat
- **CSV File**: Data tabular untuk analisis lebih lanjut
- **Grafik PNG**: Visualisasi timeline dan distribusi deteksi

### Web Dashboard Mode:
- **Real-time Dashboard**: Statistik live di web interface
- **Interactive Charts**: Timeline dan distribusi real-time
- **JSON Report**: Auto-generate saat stop deteksi
- **WebSocket Streaming**: Data real-time via WebSocket
- **System Log**: Real-time log dengan timestamp
