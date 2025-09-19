from ultralytics import YOLO
import cv2
import sys
import time
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Load YOLOv8 (nano for speed, small for accuracy)
model = YOLO("yolov8n.pt")

# Initialize reporting variables
detection_data = []
person_counts = deque(maxlen=100)  # Keep last 100 frames for real-time graph
start_time = time.time()
frame_count = 0
total_persons = 0

# Check for input argument
if len(sys.argv) > 1:
    if sys.argv[1].lower() == '--webcam' or sys.argv[1].lower() == '-w':
        cap = cv2.VideoCapture(0)  # Webcam
        print("Menggunakan input dari webcam...")
    elif sys.argv[1].startswith(('rtsp://', 'http://', 'https://')):
        # IP Camera / NVR support
        cap = cv2.VideoCapture(sys.argv[1])
        # Set buffer size to reduce latency for IP cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set timeout for network streams (5 seconds)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        print(f"Menggunakan input dari IP Camera/NVR: {sys.argv[1]}")
    else:
        cap = cv2.VideoCapture(sys.argv[1])  # File video
        print(f"Menggunakan input dari file: {sys.argv[1]}")
else:
    cap = cv2.VideoCapture("sample.mp4")  # Default
    print("Menggunakan input default: sample.mp4")

# Check if camera/video opened successfully
if not cap.isOpened():
    print("Error: Tidak dapat membuka sumber input!")
    sys.exit()

cv2.namedWindow("Detection System", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Selesai memproses atau tidak dapat membaca frame")
        break

    frame_count += 1
    current_time = time.time()
    
    # Resize for faster processing
    frame = cv2.resize(frame, (1280, 720))

    results = model(frame)
    annotated_frame = results[0].plot()

    # Count only persons (class 0)
    people_count = 0
    persons_detected = []
    
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:   # person only
            people_count += 1
            confidence = float(box.conf[0])
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            persons_detected.append({
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

    # Update statistics
    total_persons += people_count
    person_counts.append(people_count)
    
    # Store detection data for report
    detection_data.append({
        'timestamp': datetime.now().isoformat(),
        'frame': frame_count,
        'person_count': people_count,
        'persons': persons_detected,
        'elapsed_time': current_time - start_time
    })

    # Calculate real-time statistics
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_persons = total_persons / frame_count if frame_count > 0 else 0
    
    # Display real-time report
    cv2.putText(annotated_frame, f"Orang Terdeteksi: {people_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total: {total_persons} | Rata-rata: {avg_persons:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Waktu: {elapsed_time:.1f}s", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.resizeWindow("Detection System", 1920, 1080)
    cv2.imshow("Detection System", annotated_frame)

    # Print real-time console report every 30 frames
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}: {people_count} orang | Total: {total_persons} | Avg: {avg_persons:.1f} | FPS: {fps:.1f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate final report
end_time = time.time()
total_duration = end_time - start_time

print("\n" + "="*50)
print("LAPORAN DETEKSI PERSON")
print("="*50)
print(f"Total Frame: {frame_count}")
print(f"Total Durasi: {total_duration:.2f} detik")
print(f"FPS Rata-rata: {frame_count/total_duration:.2f}")
print(f"Total Person Terdeteksi: {total_persons}")
print(f"Rata-rata Person per Frame: {total_persons/frame_count:.2f}")
print(f"Max Person dalam 1 Frame: {max(person_counts) if person_counts else 0}")
print(f"Min Person dalam 1 Frame: {min(person_counts) if person_counts else 0}")

# Save to JSON file
report_filename = f"person_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_filename, 'w') as f:
    json.dump({
        'summary': {
            'total_frames': frame_count,
            'duration_seconds': total_duration,
            'average_fps': frame_count/total_duration,
            'total_persons_detected': total_persons,
            'average_persons_per_frame': total_persons/frame_count,
            'max_persons_in_frame': max(person_counts) if person_counts else 0,
            'min_persons_in_frame': min(person_counts) if person_counts else 0
        },
        'detailed_data': detection_data
    }, f, indent=2)

# Save to CSV file
csv_filename = f"person_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Timestamp', 'Person_Count', 'Elapsed_Time'])
    for data in detection_data:
        writer.writerow([data['frame'], data['timestamp'], data['person_count'], data['elapsed_time']])

# Create and save graph
plt.figure(figsize=(12, 6))
frames = [data['frame'] for data in detection_data]
counts = [data['person_count'] for data in detection_data]

plt.subplot(1, 2, 1)
plt.plot(frames, counts, 'b-', linewidth=1)
plt.title('Deteksi Person per Frame')
plt.xlabel('Frame')
plt.ylabel('Jumlah Person')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(counts, bins=20, alpha=0.7, color='green')
plt.title('Distribusi Jumlah Person')
plt.xlabel('Jumlah Person')
plt.ylabel('Frekuensi')
plt.grid(True, alpha=0.3)

plt.tight_layout()
graph_filename = f"person_detection_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nLaporan tersimpan:")
print(f"- JSON: {report_filename}")
print(f"- CSV: {csv_filename}")
print(f"- Grafik: {graph_filename}")
print("Deteksi selesai!")