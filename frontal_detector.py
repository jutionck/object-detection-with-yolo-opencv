import cv2
import numpy as np

class FrontalPersonDetector:
    def __init__(self):
        # Initialize face cascade classifier for frontal face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Alternative: Profile face cascade for side faces (to exclude)
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Minimum face size relative to person bbox
        self.min_face_ratio = 0.05  # Face should be at least 5% of person height
        self.max_face_ratio = 0.3   # Face should be at most 30% of person height
        
    def is_person_frontal(self, frame, person_bbox, confidence_threshold=0.7):
        """
        Determine if a person is facing the camera based on frontal face detection
        
        Args:
            frame: Input image frame
            person_bbox: [x1, y1, x2, y2] bounding box of detected person
            confidence_threshold: Minimum confidence for considering detection valid
            
        Returns:
            tuple: (is_frontal: bool, face_info: dict)
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
            
            # Extract person region
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                return False, {}
            
            # Convert to grayscale for face detection
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect frontal faces
            faces = self.face_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            person_height = y2 - y1
            person_width = x2 - x1
            
            frontal_faces = []
            for (fx, fy, fw, fh) in faces:
                # Check if face size is reasonable relative to person
                face_ratio = fh / person_height
                
                if self.min_face_ratio <= face_ratio <= self.max_face_ratio:
                    # Check face position (should be in upper portion of person)
                    face_y_ratio = fy / person_height
                    
                    if face_y_ratio <= 0.5:  # Face in upper 50% of person
                        # Calculate face quality metrics
                        face_roi = gray_roi[fy:fy+fh, fx:fx+fw]
                        face_quality = self._assess_face_quality(face_roi)
                        
                        if face_quality > confidence_threshold:
                            frontal_faces.append({
                                'bbox': [fx + x1, fy + y1, fw, fh],
                                'quality': face_quality,
                                'face_ratio': face_ratio,
                                'position_ratio': face_y_ratio
                            })
            
            # Check for profile faces (to exclude)
            profile_faces = self.profile_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
            
            has_profile = len(profile_faces) > 0
            has_frontal = len(frontal_faces) > 0
            
            # Person is frontal if:
            # 1. Has frontal face(s) detected
            # 2. No strong profile face detected OR frontal face quality is higher
            is_frontal = has_frontal and (not has_profile or len(frontal_faces) >= len(profile_faces))
            
            face_info = {
                'frontal_faces': frontal_faces,
                'profile_faces_count': len(profile_faces),
                'best_frontal_quality': max([f['quality'] for f in frontal_faces]) if frontal_faces else 0,
                'person_size': (person_width, person_height)
            }
            
            return is_frontal, face_info
            
        except Exception as e:
            print(f"Error in frontal detection: {e}")
            return False, {}
    
    def _assess_face_quality(self, face_roi):
        """
        Assess the quality of detected face for frontal determination
        
        Args:
            face_roi: Grayscale face region
            
        Returns:
            float: Quality score (0-1)
        """
        if face_roi.size == 0:
            return 0.0
        
        try:
            # Calculate face quality metrics
            
            # 1. Contrast (higher is better for clear faces)
            contrast = face_roi.std() / 255.0
            
            # 2. Symmetry check (frontal faces should be more symmetric)
            h, w = face_roi.shape
            left_half = face_roi[:, :w//2]
            right_half = cv2.flip(face_roi[:, w//2:], 1)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate symmetry score
            if left_half.shape == right_half.shape:
                symmetry = 1.0 - (np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0)
            else:
                symmetry = 0.5
            
            # 3. Edge density (faces should have reasonable edge content)
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / (face_roi.shape[0] * face_roi.shape[1])
            edge_score = min(edge_density * 10, 1.0)  # Normalize
            
            # Combine metrics
            quality_score = (contrast * 0.3 + symmetry * 0.5 + edge_score * 0.2)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Error in face quality assessment: {e}")
            return 0.0
    
    def annotate_frontal_persons(self, frame, person_detections, frontal_results):
        """
        Annotate frame with frontal person indicators
        
        Args:
            frame: Input frame
            person_detections: List of person detection results
            frontal_results: List of frontal detection results
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = frame.copy()
        
        for i, (person_bbox, (is_frontal, face_info)) in enumerate(zip(person_detections, frontal_results)):
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox['bbox']]
            
            # Color coding: Green for frontal, Red for non-frontal
            color = (0, 255, 0) if is_frontal else (0, 0, 255)
            thickness = 3 if is_frontal else 2
            
            # Draw person bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"Frontal Person" if is_frontal else "Person"
            if 'confidence' in person_bbox:
                label += f" {person_bbox['confidence']:.2f}"
                
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw detected faces
            if is_frontal and face_info.get('frontal_faces'):
                for face in face_info['frontal_faces']:
                    fx, fy, fw, fh = face['bbox']
                    cv2.rectangle(annotated_frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Face {face['quality']:.2f}", 
                               (fx, fy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated_frame
    
    def filter_frontal_persons(self, frame, yolo_results):
        """
        Filter YOLO person detections to keep only frontal-facing persons
        
        Args:
            frame: Input frame
            yolo_results: YOLO detection results
            
        Returns:
            tuple: (frontal_persons, all_persons, frontal_results)
        """
        all_persons = []
        frontal_persons = []
        frontal_results = []
        
        # Extract person detections (class 0)
        for box in yolo_results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Person class
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                person_data = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence
                }
                all_persons.append(person_data)
                
                # Check if person is frontal
                is_frontal, face_info = self.is_person_frontal(frame, [x1, y1, x2, y2])
                frontal_results.append((is_frontal, face_info))
                
                if is_frontal:
                    person_data['face_info'] = face_info
                    frontal_persons.append(person_data)
        
        return frontal_persons, all_persons, frontal_results