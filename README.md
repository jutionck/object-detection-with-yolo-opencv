# Sistem Deteksi Wajah dengan OpenCV Haar Cascades

Sistem deteksi wajah yang dibangun menggunakan OpenCV Haar Cascades, mampu mendeteksi dan menganalisis wajah dalam gambar, video dan kamera secara real-time dengan analisis demografis (gender dan usia).

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

## 📦 Dependensi

### Core Dependencies:
- Python 3.8+
- OpenCV (4.10.0.84) - untuk deteksi wajah dengan Haar Cascades
- NumPy (1.26.4) - operasi array dan matematika
- Matplotlib (>=3.5.0) - visualisasi grafik dan laporan

### Web Dashboard Dependencies:
- Flask (>=2.3.0) - web framework
- Flask-SocketIO (>=5.3.0) - real-time communication
- Python-SocketIO (>=5.8.0) - WebSocket support

### Install manual jika diperlukan:

```bash
# Core dependencies
pip install opencv-python numpy matplotlib

# Web dependencies
pip install flask flask-socketio python-socketio
```

## ▶️ Penggunaan

Sistem memiliki 2 mode operasi:

### 🖥️ **Mode Console (main.py)**

**Jalankan dengan video default:**
```bash
python3 main.py
```

**Jalankan dengan webcam:**
```bash
python3 main.py --webcam
# atau
python3 main.py -w
```

**Jalankan dengan file video lain:**
```bash
python3 main.py path/ke/video.mp4
```

**Jalankan dengan IP Camera/NVR:**
```bash
# RTSP stream
python3 main.py rtsp://admin:password@192.168.1.100:554/stream

# HTTP stream
python3 main.py http://192.168.1.100:8080/video.cgi
```

Sistem akan menampilkan deteksi wajah real-time dengan analisis demografis. Tekan 'q' untuk keluar dan menghasilkan laporan akhir.

### 🌐 **Mode Web Dashboard (app.py)**

**Jalankan web dashboard:**
```bash
# Recommended - menggunakan launcher
python3 run_web.py

# Direct - langsung Flask app
python3 app.py
```

**Akses dashboard:**
Buka browser: **http://localhost:9000**

**Fitur Web Dashboard:**
- 🎥 **Live Video Feed**: Stream video real-time dengan deteksi wajah
- 📊 **Real-time Statistics**: FPS, jumlah wajah, rata-rata, demografis
- 📈 **Interactive Charts**: Timeline deteksi & distribusi wajah
- 📊 **Demographics Analysis**: Gender dan kelompok usia real-time
- 🎮 **Interactive Controls**: Start/stop, pilih sumber input
- 📱 **Responsive Design**: Desktop & mobile friendly
- 🔌 **Multiple Input Sources**: 
  - Video default (`sample.mp4`)
  - Webcam
  - IP Camera/NVR (RTSP/HTTP)

## 👤 **Fitur Deteksi Wajah:**

### 🎯 **Algoritma Deteksi:**
- **OpenCV Haar Cascades**: Deteksi wajah cepat dan akurat
- **Multiple Cascade Support**: Frontal, profile, dan alternative cascades
- **Quality Assessment**: Penilaian kualitas wajah (blur, brightness, contrast)
- **Duplicate Removal**: Eliminasi deteksi ganda dengan IoU threshold

### 🎭 **Mode Performa:**
- **Fast Mode**: Deteksi frontal saja, optimal untuk real-time
- **Balanced Mode**: Frontal + profile, keseimbangan speed-accuracy
- **Quality Mode**: Semua cascade + full quality assessment

### 📊 **Analisis Demografis:**
- **Gender Detection**: Klasifikasi laki-laki/perempuan (heuristik)
- **Age Estimation**: Kelompok usia (0-12, 15-32, 38-53, 60+)
- **Face Quality Scoring**: Skor kualitas berdasarkan multiple metrics
- **Real-time Analytics**: Update statistik demografis secara langsung

### 🎨 **Visualisasi:**
- **Bounding Boxes**: Kotak deteksi dengan warna berdasarkan kualitas
- **Demographics Labels**: Label gender dan usia pada setiap wajah
- **Quality Indicators**: Indikator tipe deteksi (frontal/profile/alternative)
- **Real-time Stats Overlay**: Informasi statistik langsung di video

## 📋 **Output Laporan:**

### Console Mode:
- **Real-time Display**: Statistik wajah langsung di layar video
- **Console Log**: Laporan berkala setiap 30 frame dengan demografis
- **JSON Report**: `face_detection_report_YYYYMMDD_HHMMSS.json`
  - Data lengkap deteksi dengan koordinat dan demografis
  - Summary statistik dan distribusi gender/usia
- **CSV File**: `face_detection_data_YYYYMMDD_HHMMSS.csv`
  - Data tabular per frame untuk analisis
  - Kolom: Frame, Timestamp, Face_Count, Males, Females, Unknown
- **Grafik PNG**: `face_detection_graph_YYYYMMDD_HHMMSS.png`
  - Timeline deteksi wajah
  - Distribusi gender dan kelompok usia

### Web Dashboard Mode:
- **Real-time Dashboard**: Statistik wajah live di web interface
- **Interactive Charts**: Timeline deteksi dan distribusi wajah
- **Demographics Dashboard**: Real-time gender dan age distribution
- **JSON Report**: `web_detection_report_YYYYMMDD_HHMMSS.json`
- **WebSocket Streaming**: Data real-time via WebSocket
- **System Log**: Real-time log deteksi dengan demografis

## ⚙️ **Spesifikasi Teknis:**

### 🚀 **Performa:**
- **Frame Rate**: 15-30+ FPS (tergantung hardware dan mode)
- **Resolution Support**: 480p - 4K (otomatis resize ke 1280x720 untuk processing)
- **Memory Usage**: ~200-500MB RAM (tanpa YOLO dependencies)
- **CPU Usage**: Moderate (Haar Cascades lebih ringan dari deep learning)

### 📱 **Input Sources Supported:**
- **Local Files**: .mp4, .avi, .mov, .mkv, dll
- **Webcam**: USB camera, built-in laptop camera
- **IP Camera**: RTSP streams (Hikvision, Dahua, dll)
- **NVR Systems**: Network Video Recorder streams
- **HTTP Streams**: MJPEG dan HTTP video streams

### 🎯 **Akurasi & Limitasi:**
- **Face Detection**: Good accuracy untuk frontal faces (80-95%)
- **Profile Detection**: Moderate accuracy untuk side faces (60-80%)
- **Demographics**: Heuristic-based (akurasi ~60-70%, untuk demo purposes)
- **Optimal Conditions**: Pencahayaan baik, ukuran wajah >30px
- **Challenges**: Pencahayaan buruk, wajah kecil, oklusi partial

## 🛠️ **Troubleshooting:**

### Common Issues:

**⚠️ Kamera tidak terdeteksi:**
```bash
# Cek kamera tersedia
python3 -c "import cv2; print('Camera available:', cv2.VideoCapture(0).isOpened())"
```

**⚠️ IP Camera tidak connect:**
- Pastikan URL RTSP benar
- Cek network connectivity
- Verifikasi username/password
- Test dengan VLC media player dulu

**⚠️ Performance lambat:**
- Gunakan mode 'fast' untuk webcam
- Kurangi resolusi input
- Close aplikasi lain yang menggunakan camera

**⚠️ Web dashboard tidak load:**
```bash
# Cek port 9000 tersedia
lsof -i :9000

# Ganti port jika perlu
FLASK_PORT=8080 python3 run_web.py
```

## 📝 **Changelog & Migration:**

### v2.0.0 - Face Detection Update
- ⚡ **BREAKING CHANGE**: Migrasi dari YOLOv8 person detection ke OpenCV face detection
- ➕ **Added**: Face detection dengan Haar Cascades
- ➕ **Added**: Demographics analysis (gender/age)
- ➕ **Added**: Multiple performance modes
- ➕ **Added**: Face quality assessment
- ➖ **Removed**: YOLOv8, PyTorch dependencies
- 🔄 **Changed**: Output file formats dan API
- 🔄 **Changed**: Web dashboard untuk face-focused metrics

### Migration dari v1.x:
1. Update dependencies: `pip install -r requirements.txt`
2. File output berubah dari `person_detection_*` ke `face_detection_*`
3. API statistics berubah (lihat code untuk details)
4. Tidak ada backward compatibility dengan format lama

## 🚀 **Quick Start Example:**

```bash
# Clone dan setup
git clone https://github.com/jutionck/object-detection-with-yolo-opencv.git
cd object-detection-with-yolo-opencv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test dengan webcam
python3 main.py --webcam

# Atau jalankan web dashboard
python3 run_web.py
# Buka http://localhost:9000
```

## 🤝 **Contributing:**

Kontribusi sangat welcome! Silakan:
1. Fork repository ini
2. Buat branch feature (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 **License:**

MIT License - lihat file [LICENSE](LICENSE) untuk details.

## 👤 **Author:**

**Jutionck** - [GitHub Profile](https://github.com/jutionck)

## 🙏 **Acknowledgments:**

- OpenCV team untuk Haar Cascades algorithms
- Flask dan SocketIO communities
- Chart.js untuk interactive visualizations
- Semua contributors dan testers

---

🌟 **Jika project ini membantu, berikan star di GitHub!** 🌟
