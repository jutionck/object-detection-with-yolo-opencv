# 🌐 Web Dashboard untuk Real-time Object Detection

Dashboard web real-time untuk sistem deteksi objek menggunakan YOLOv8, Flask, dan WebSocket.

## 🚀 Fitur Web Dashboard

### 📊 Real-time Monitoring
- **Live Video Feed**: Stream video real-time dengan bounding boxes
- **Statistik Real-time**: Jumlah orang saat ini, total, rata-rata, FPS
- **Grafik Timeline**: Visualisasi deteksi sepanjang waktu
- **Distribusi Data**: Histogram frekuensi jumlah orang terdeteksi
- **Log Real-time**: Catatan deteksi dengan timestamp

### 🎮 Kontrol Interaktif
- **Start/Stop Detection**: Kontrol deteksi dengan tombol
- **Pilihan Input**: Video default, webcam, atau file custom
- **Responsive Design**: Tampilan optimal di desktop dan mobile

### 📁 Output Otomatis
- **JSON Report**: Data lengkap dengan timestamp dan koordinat
- **Real-time Updates**: Data streaming via WebSocket

## 🛠️ Setup Web Dashboard

### 1. Install Dependencies
```bash
pip install -r Requirements.txt
```

Dependencies tambahan untuk web:
- Flask (>= 2.3.0)
- Flask-SocketIO (>= 5.3.0) 
- Python-SocketIO (>= 5.8.0)

### 2. Jalankan Web Server
```bash
# Cara 1: Menggunakan script launcher
python run_web.py

# Cara 2: Langsung menjalankan Flask app
python app.py
```

### 3. Akses Dashboard
Buka browser dan akses: **http://localhost:5000**

## 📱 Cara Menggunakan Web Dashboard

### 1. **Koneksi Status**
- 🟢 **Terhubung**: WebSocket connection berhasil
- 🔵 **Deteksi Berjalan**: Sistem sedang memproses video
- 🔴 **Tidak Terhubung**: Connection error

### 2. **Memulai Deteksi**
1. Pilih sumber input:
   - **Video Default**: Menggunakan `sample.mp4`
   - **Webcam**: Menggunakan kamera sistem
2. Klik **"Mulai Deteksi"**
3. Dashboard akan menampilkan:
   - Live video feed dengan bounding boxes
   - Statistik real-time
   - Grafik timeline dan distribusi
   - Log deteksi

### 3. **Menghentikan Deteksi**
1. Klik **"Stop Deteksi"**
2. Sistem akan generate laporan JSON otomatis
3. File tersimpan dengan nama: `web_detection_report_[timestamp].json`

## 🎯 Fitur Dashboard

### 📊 Panel Statistik Real-time
- **Orang Saat Ini**: Jumlah deteksi pada frame current
- **Total Terdeteksi**: Akumulasi deteksi selama sesi
- **Rata-rata**: Rata-rata deteksi per frame
- **FPS**: Frame rate pemrosesan
- **Frame Count**: Total frame yang diproses
- **Waktu Berjalan**: Durasi deteksi

### 📈 Visualisasi Data
- **Timeline Chart**: Grafik line menampilkan jumlah deteksi per frame
- **Distribution Chart**: Histogram frekuensi jumlah orang terdeteksi
- **Real-time Updates**: Chart ter-update otomatis setiap frame

### 📝 System Log
- Log real-time setiap deteksi dengan timestamp
- Format: `[HH:MM:SS] Frame XXX: Y orang terdeteksi`
- Auto-scroll dengan limit 100 entries terakhir

## 🏗️ Arsitektur Web System

### Backend (Flask + SocketIO)
```
app.py
├── DetectionSystem Class
│   ├── Video capture & processing
│   ├── YOLO inference
│   └── Real-time statistics
├── Flask Routes
│   ├── /api/start - Mulai deteksi
│   ├── /api/stop - Stop deteksi
│   └── /api/status - Status sistem
└── WebSocket Events
    ├── detection_update - Stream data real-time
    ├── connect/disconnect - Connection handling
    └── Real-time frame streaming
```

### Frontend (HTML + JavaScript + Chart.js)
```
templates/index.html
├── Real-time Video Display
├── Statistics Dashboard
├── Interactive Controls
├── Chart.js Visualization
│   ├── Timeline Chart (Line)
│   └── Distribution Chart (Bar)
└── WebSocket Client
    ├── Real-time data handling
    ├── Chart updates
    └── UI state management
```

## 🔧 Konfigurasi

### Server Configuration
- **Host**: `0.0.0.0` (accessible from network)
- **Port**: `5000`
- **Debug Mode**: Enabled (development)

### Video Processing
- **Resolution**: 1280x720 (optimized for speed)
- **Model**: YOLOv8 nano (`yolov8n.pt`)
- **Target Class**: Person (class 0)
- **Frame Rate**: ~30 FPS

### WebSocket Settings
- **CORS**: Allowed from all origins
- **Real-time Updates**: Every frame (~33ms)
- **Data Format**: JSON with base64 encoded frames

## 🚨 Troubleshooting

### Common Issues

**1. Import Error (flask/flask_socketio)**
```bash
pip install flask flask-socketio
```

**2. Video Source Error**
- Pastikan `sample.mp4` exists atau gunakan webcam
- Check camera permissions untuk webcam access

**3. YOLO Model Missing**
- Model `yolov8n.pt` akan auto-download pada first run
- Pastikan internet connection untuk download

**4. Port Already in Use**
```bash
# Kill existing process on port 5000
lsof -ti:5000 | xargs kill -9
```

### Performance Tips

1. **Video Resolution**: Default 1280x720 optimal untuk speed
2. **Frame Rate**: Adjust delay di detection loop untuk performance
3. **Chart Data**: Limit timeline chart ke 50 points terakhir
4. **Browser**: Gunakan Chrome/Firefox untuk WebSocket support optimal

## 📊 Output Files

### JSON Report Structure
```json
{
  "summary": {
    "total_frames": 1500,
    "duration_seconds": 50.23,
    "average_fps": 29.86,
    "total_persons_detected": 450,
    "average_persons_per_frame": 3.2,
    "max_persons_in_frame": 8,
    "min_persons_in_frame": 0
  },
  "detailed_data": [
    {
      "timestamp": "2024-01-15T10:30:45.123",
      "frame": 1,
      "person_count": 3,
      "persons": [
        {
          "confidence": 0.85,
          "bbox": [120, 50, 200, 300]
        }
      ],
      "elapsed_time": 0.033
    }
  ]
}
```

## 🔗 Integration dengan Main System

Web dashboard bisa dijalankan bersamaan dengan `main.py`:
- **main.py**: Console-based detection dengan output files
- **app.py**: Web-based real-time dashboard
- Kedua sistem menggunakan model dan pipeline yang sama

## 🎨 Customization

### Styling
- Edit `templates/index.html` untuk mengubah tampilan
- CSS menggunakan CSS Grid dan Flexbox
- Responsive design dengan media queries

### Features
- Tambah chart types di Chart.js configuration
- Extend API endpoints untuk features tambahan
- Modify detection classes (tidak hanya person)

---

🎯 **Web Dashboard** memberikan experience real-time monitoring yang interaktif untuk sistem deteksi objek, dengan visualisasi data yang comprehensive dan kontrol yang user-friendly.