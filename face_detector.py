import cv2
import numpy as np
from gender_age_detector import gender_age_detector

class FaceDetector:
    def __init__(self, performance_mode='balanced'):
        """
        Face detection system using OpenCV Haar Cascades
        
        performance_mode:
        - 'fast': Only frontal face detection, minimal quality checks
        - 'balanced': Frontal + profile, basic quality checks  
        - 'quality': All cascades with full quality assessment
        """
        self.performance_mode = performance_mode
        
        # Always load frontal face cascade
        self.frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load additional cascades based on performance mode
        if performance_mode in ['balanced', 'quality']:
            self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        if performance_mode == 'quality':
            self.alt_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            self.alt2_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        
        # Adjust thresholds based on performance mode
        if performance_mode == 'fast':
            self.confidence_threshold = 0.3
            self.min_neighbors = 3
            self.scale_factor = 1.1
            self.min_size = (20, 20)
        elif performance_mode == 'balanced':
            self.confidence_threshold = 0.4
            self.min_neighbors = 4
            self.scale_factor = 1.05
            self.min_size = (25, 25)
        else:  # quality
            self.confidence_threshold = 0.5
            self.min_neighbors = 5
            self.scale_factor = 1.03
            self.min_size = (30, 30)
        
        # Frame skip for performance optimization
        self.frame_skip = 1 if performance_mode == 'quality' else 2 if performance_mode == 'balanced' else 3
        self.frame_counter = 0
        
    def detect_faces(self, frame):
        """
        Detect faces in the entire frame
        
        Args:
            frame: Input frame
            
        Returns:
            list: List of detected faces with their information
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            if self.performance_mode == 'quality':
                gray = cv2.equalizeHist(gray)
            
            all_faces = []
            
            # Detect frontal faces (always enabled)
            frontal_faces = self.frontal_face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in frontal_faces:
                all_faces.append((x, y, w, h, 'frontal'))
            
            # Detect profile faces (if enabled)
            if hasattr(self, 'profile_face_cascade'):
                profile_faces = self.profile_face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor + 0.05,
                    minNeighbors=self.min_neighbors - 1,
                    minSize=self.min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in profile_faces:
                    all_faces.append((x, y, w, h, 'profile'))
            
            # Detect with alternative cascades (if enabled)
            if hasattr(self, 'alt_face_cascade'):
                alt_faces = self.alt_face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor + 0.05,
                    minNeighbors=self.min_neighbors - 1,
                    minSize=self.min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in alt_faces:
                    all_faces.append((x, y, w, h, 'alternative'))
            
            if hasattr(self, 'alt2_face_cascade'):
                alt2_faces = self.alt2_face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor + 0.05,
                    minNeighbors=self.min_neighbors - 1,
                    minSize=self.min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (x, y, w, h) in alt2_faces:
                    all_faces.append((x, y, w, h, 'alternative2'))
            
            # Remove duplicates
            if self.performance_mode == 'fast':
                unique_faces = all_faces  # Skip duplicate removal for speed
            else:
                unique_faces = self._remove_duplicate_faces(all_faces)
            
            # Process detected faces
            processed_faces = []
            for (x, y, w, h, face_type) in unique_faces:
                # Assess face quality
                face_roi = gray[y:y+h, x:x+w]
                quality = self._assess_face_quality(face_roi)
                
                if quality > self.confidence_threshold:
                    face_data = {
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'quality': float(quality),
                        'type': str(face_type),
                        'area': int(w * h)
                    }
                    
                    # Add demographics analysis
                    demographics = gender_age_detector.analyze_face_demographics(frame, [x, y, w, h])
                    face_data['demographics'] = demographics
                    
                    processed_faces.append(face_data)
            
            # Sort by quality (best faces first)
            processed_faces.sort(key=lambda x: x['quality'], reverse=True)
            
            return processed_faces
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def _remove_duplicate_faces(self, all_faces, overlap_threshold=0.3):
        """
        Remove duplicate face detections with overlap threshold
        """
        if not all_faces:
            return []
        
        # Sort by area (larger faces first)
        all_faces.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        unique_faces = []
        for face in all_faces:
            x1, y1, w1, h1, face_type = face
            is_duplicate = False
            
            for existing_face in unique_faces:
                x2, y2, w2, h2, _ = existing_face
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0:
                    iou = overlap_area / union_area
                    if iou > overlap_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _assess_face_quality(self, face_roi):
        """
        Assess the quality of detected face
        """
        if face_roi.size == 0:
            return 0.0
        
        try:
            # Calculate contrast
            contrast = face_roi.std() / 255.0
            
            # Calculate brightness
            mean_brightness = np.mean(face_roi) / 255.0
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            
            # Size check
            h, w = face_roi.shape
            size_score = 1.0 if min(h, w) >= 30 else min(h, w) / 30.0
            
            # Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 100.0, 1.0)
            
            # Combined quality score
            quality_score = (
                contrast * 0.3 + 
                brightness_score * 0.2 + 
                size_score * 0.3 + 
                blur_score * 0.2
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception:
            return 0.0
    
    def annotate_faces(self, frame, faces):
        """
        Draw bounding boxes and labels on detected faces
        """
        annotated_frame = frame.copy()
        
        for i, face in enumerate(faces):
            x, y, w, h = face['bbox']
            
            # Color coding based on quality
            if face['quality'] > 0.7:
                color = (0, 255, 0)  # Green for high quality
            elif face['quality'] > 0.5:
                color = (0, 255, 255)  # Yellow for medium quality
            else:
                color = (0, 165, 255)  # Orange for lower quality
            
            # Draw face bounding box
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw face label
            label = f"Face {i+1}"
            if 'demographics' in face and face['demographics']:
                demo = face['demographics']
                gender = demo.get('gender', 'Unknown')
                age = demo.get('age_group', 'Unknown')
                label += f" | {gender[:1]},{age}"
            
            label += f" ({face['quality']:.2f})"
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = y - 10 if y - 10 > label_size[1] else y + h + 20
            
            cv2.putText(annotated_frame, label, (x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw face type indicator
            type_label = face['type']
            cv2.putText(annotated_frame, type_label, (x + w - 40, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return annotated_frame
    
    def should_process_frame(self):
        """
        Frame skip logic for performance optimization
        """
        self.frame_counter += 1
        return self.frame_counter % self.frame_skip == 0