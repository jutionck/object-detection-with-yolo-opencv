import cv2
import sys
import time
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from face_detector import FaceDetector
from gender_age_detector import gender_age_detector

# Initialize face detector with balanced mode for console
face_detector = FaceDetector(performance_mode='balanced')

# Initialize reporting variables
detection_data = []
face_counts = deque(maxlen=100)  # Keep last 100 frames for real-time graph
start_time = time.time()
frame_count = 0
total_faces = 0

# Demographics tracking
demographics_history = []
gender_stats = {'Male': 0, 'Female': 0, 'Unknown': 0}
age_stats = {'child': 0, 'young': 0, 'adult': 0, 'senior': 0}

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

    # Detect faces directly
    faces = face_detector.detect_faces(frame)
    
    # Create annotated frame with face indicators
    annotated_frame = face_detector.annotate_faces(frame, faces)

    # Count faces
    face_count = len(faces)
    
    faces_detected = []
    
    # Process faces and collect demographics
    current_frame_demographics = []
    for face in faces:
        face_data = {
            'quality': float(face['quality']),
            'bbox': [float(x) for x in face['bbox']],
            'type': face['type'],
            'area': int(face['area'])
        }
        
        # Add demographics if available
        if 'demographics' in face:
            face_data['demographics'] = face['demographics']
            current_frame_demographics.append(face['demographics'])
            
            # Update statistics
            demo = face['demographics']
            if demo.get('gender') in gender_stats:
                gender_stats[demo['gender']] += 1
            else:
                gender_stats['Unknown'] += 1
            
            # Update age statistics
            age_group = demo.get('age_group', '')
            if any(age in age_group for age in ['0-2', '4-6', '8-12']):
                age_stats['child'] += 1
            elif any(age in age_group for age in ['15-20', '25-32']):
                age_stats['young'] += 1
            elif any(age in age_group for age in ['38-43', '48-53']):
                age_stats['adult'] += 1
            elif '60-100' in age_group:
                age_stats['senior'] += 1
        
        faces_detected.append(face_data)
    
    # Store demographics for this frame
    if current_frame_demographics:
        demographics_history.extend(current_frame_demographics)

    # Update statistics
    total_faces += face_count
    face_counts.append(face_count)
    
    # Store detection data for report
    detection_data.append({
        'timestamp': datetime.now().isoformat(),
        'frame': frame_count,
        'face_count': face_count,
        'faces': faces_detected,
        'elapsed_time': current_time - start_time
    })

    # Calculate real-time statistics
    elapsed_time = current_time - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    avg_faces = total_faces / frame_count if frame_count > 0 else 0
    
    # Display real-time report
    cv2.putText(annotated_frame, f"Wajah Terdeteksi: {face_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total Wajah: {total_faces}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Rata-rata Wajah: {avg_faces:.1f}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Waktu: {elapsed_time:.1f}s", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.resizeWindow("Detection System", 1920, 1080)
    cv2.imshow("Detection System", annotated_frame)

    # Print real-time console report every 30 frames
    if frame_count % 30 == 0:
        # Demographics summary for current session
        demo_summary = ""
        if current_frame_demographics:
            males = sum(1 for d in current_frame_demographics if d.get('gender') == 'Male')
            females = sum(1 for d in current_frame_demographics if d.get('gender') == 'Female')
            demo_summary = f" | Demo: {males}M/{females}F"
        
        print(f"Frame {frame_count}: Wajah Terdeteksi={face_count} | Total: {total_faces} | FPS: {fps:.1f}{demo_summary}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate final report
end_time = time.time()
total_duration = end_time - start_time

print("\n" + "="*50)
print("LAPORAN DETEKSI WAJAH")
print("="*50)
print(f"Total Frame: {frame_count}")
print(f"Total Durasi: {total_duration:.2f} detik")
print(f"FPS Rata-rata: {frame_count/total_duration:.2f}")
print(f"Total Wajah Terdeteksi: {total_faces}")
print(f"Rata-rata Wajah per Frame: {total_faces/frame_count:.2f}")
print(f"Max Wajah dalam 1 Frame: {max(face_counts) if face_counts else 0}")
print(f"Min Wajah dalam 1 Frame: {min(face_counts) if face_counts else 0}")

# Demographics Summary
print("\n" + "="*30)
print("DEMOGRAFIS SUMMARY")
print("="*30)
total_demographics = sum(gender_stats.values())
if total_demographics > 0:
    print(f"Total Wajah dengan Demografis: {total_demographics}")
    print("\nGender Distribution:")
    for gender, count in gender_stats.items():
        percentage = (count / total_demographics) * 100
        print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    print("\nAge Group Distribution:")
    total_age = sum(age_stats.values())
    for age_group, count in age_stats.items():
        percentage = (count / total_age) * 100 if total_age > 0 else 0
        print(f"  {age_group.title()}: {count} ({percentage:.1f}%)")
else:
    print("Tidak ada data demografis yang terdeteksi")

# Save to JSON file
report_filename = f"face_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
# Calculate demographics percentages
total_demographics = sum(gender_stats.values())
gender_percentages = {}
age_percentages = {}

if total_demographics > 0:
    for gender, count in gender_stats.items():
        gender_percentages[gender] = round((count / total_demographics) * 100, 1)
    
    total_age = sum(age_stats.values())
    if total_age > 0:
        for age_group, count in age_stats.items():
            age_percentages[age_group] = round((count / total_age) * 100, 1)

with open(report_filename, 'w') as f:
    json.dump({
        'summary': {
            'total_frames': frame_count,
            'duration_seconds': total_duration,
            'average_fps': frame_count/total_duration,
            'total_faces_detected': total_faces,
            'average_faces_per_frame': total_faces/frame_count,
            'max_faces_in_frame': max(face_counts) if face_counts else 0,
            'min_faces_in_frame': min(face_counts) if face_counts else 0,
            'demographics': {
                'total_analyzed': total_demographics,
                'gender_distribution': gender_stats,
                'gender_percentages': gender_percentages,
                'age_distribution': age_stats,
                'age_percentages': age_percentages
            }
        },
        'detailed_data': detection_data,
        'demographics_history': demographics_history
    }, f, indent=2)

# Save to CSV file
csv_filename = f"face_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'Timestamp', 'Face_Count', 'Elapsed_Time', 'Males_Detected', 'Females_Detected', 'Unknown_Gender'])
    for data in detection_data:
        # Count demographics for this frame
        frame_males = 0
        frame_females = 0
        frame_unknown = 0
        
        for face in data.get('faces', []):
            if 'demographics' in face:
                gender = face['demographics'].get('gender', 'Unknown')
                if gender == 'Male':
                    frame_males += 1
                elif gender == 'Female':
                    frame_females += 1
                else:
                    frame_unknown += 1
        
        writer.writerow([data['frame'], data['timestamp'], data['face_count'], 
                        data['elapsed_time'], frame_males, frame_females, frame_unknown])

# Create and save graph
plt.figure(figsize=(16, 12))
frames = [data['frame'] for data in detection_data]
face_counts_data = [data['face_count'] for data in detection_data]

# Timeline plot
plt.subplot(2, 2, 1)
plt.plot(frames, face_counts_data, 'g-', linewidth=1, label='Wajah Terdeteksi')
plt.title('Deteksi Wajah per Frame')
plt.xlabel('Frame')
plt.ylabel('Jumlah Wajah')
plt.legend()
plt.grid(True, alpha=0.3)

# Distribution - Face counts
plt.subplot(2, 2, 2)
plt.hist(face_counts_data, bins=20, alpha=0.7, color='green', label='Wajah')
plt.title('Distribusi Jumlah Wajah')
plt.xlabel('Jumlah Wajah')
plt.ylabel('Frekuensi')
plt.grid(True, alpha=0.3)

# Gender distribution pie chart
plt.subplot(2, 2, 3)
if total_demographics > 0 and any(gender_stats.values()):
    gender_labels = []
    gender_values = []
    gender_colors = ['#87CEEB', '#FFB6C1', '#D3D3D3']  # Light blue, light pink, light gray
    
    for i, (gender, count) in enumerate(gender_stats.items()):
        if count > 0:
            gender_labels.append(f'{gender}\n({count})')
            gender_values.append(count)
    
    plt.pie(gender_values, labels=gender_labels, autopct='%1.1f%%', 
            colors=gender_colors[:len(gender_values)], startangle=90)
    plt.title('Distribusi Gender')
else:
    plt.text(0.5, 0.5, 'Tidak ada data\ndemografis', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Distribusi Gender')

# Age group distribution bar chart  
plt.subplot(2, 2, 4)
if total_demographics > 0 and any(age_stats.values()):
    age_labels = []
    age_values = []
    
    for age_group, count in age_stats.items():
        if count > 0:
            age_labels.append(age_group.title())
            age_values.append(count)
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    plt.bar(age_labels, age_values, color=colors[:len(age_values)], alpha=0.7)
    plt.title('Distribusi Kelompok Usia')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(age_values):
        plt.text(i, v + max(age_values)*0.01, str(v), ha='center', va='bottom')
else:
    plt.text(0.5, 0.5, 'Tidak ada data demografis', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Distribusi Kelompok Usia')

plt.tight_layout()
graph_filename = f"face_detection_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nLaporan tersimpan:")
print(f"- JSON: {report_filename}")
print(f"- CSV: {csv_filename}")
print(f"- Grafik: {graph_filename}")
print("Deteksi selesai!")