import cv2
import numpy as np
from gender_age_detector import gender_age_detector

class FrontalPersonDetector:
    def __init__(self, performance_mode='balanced'):
        """
        Lite version of frontal person detector for better performance
        
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
        
        # Adjust thresholds based on performance mode
        if performance_mode == 'fast':
            self.min_face_ratio = 0.03  # More lenient
            self.max_face_ratio = 0.4   # More lenient  
            self.confidence_threshold = 0.3  # Much lower threshold
            self.min_neighbors = 2      # More lenient
        elif performance_mode == 'balanced':
            self.min_face_ratio = 0.03
            self.max_face_ratio = 0.4
            self.confidence_threshold = 0.4  # Lower threshold
            self.min_neighbors = 3      # More lenient
        else:  # quality
            self.min_face_ratio = 0.03
            self.max_face_ratio = 0.4
            self.confidence_threshold = 0.3  # Lower threshold
            self.min_neighbors = 4      # More lenient
        
        # Frame skip for performance optimization
        self.frame_skip = 1 if performance_mode == 'quality' else 2 if performance_mode == 'balanced' else 3
        self.frame_counter = 0
        self.last_detection_results = []
        
    def is_person_frontal(self, frame, person_bbox, confidence_threshold=None):
        """
        Optimized face visibility detection
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        try:
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
            
            # Extract person region
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                return False, {}
            
            # Convert to grayscale for face detection
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection in poor lighting
            if self.performance_mode == 'quality':
                gray_roi = cv2.equalizeHist(gray_roi)
            
            all_faces = []
            
            # Detect frontal faces (always enabled)
            frontal_faces = self.frontal_face_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.05,  # Smaller scale factor for better detection
                minNeighbors=self.min_neighbors,
                minSize=(10, 10),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (fx, fy, fw, fh) in frontal_faces:
                all_faces.append((fx, fy, fw, fh, 'frontal'))
            
            # Detect profile faces (if enabled)
            if hasattr(self, 'profile_face_cascade'):
                profile_faces = self.profile_face_cascade.detectMultiScale(
                    gray_roi,
                    scaleFactor=1.15,  # Slightly larger scale factor for performance
                    minNeighbors=self.min_neighbors - 1,
                    minSize=(15, 15),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (fx, fy, fw, fh) in profile_faces:
                    all_faces.append((fx, fy, fw, fh, 'profile'))
            
            # Detect with alternative cascade (if enabled)
            if hasattr(self, 'alt_face_cascade'):
                alt_faces = self.alt_face_cascade.detectMultiScale(
                    gray_roi,
                    scaleFactor=1.15,
                    minNeighbors=self.min_neighbors - 1,
                    minSize=(15, 15),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (fx, fy, fw, fh) in alt_faces:
                    all_faces.append((fx, fy, fw, fh, 'alternative'))
            
            # Remove duplicates (simplified for performance)
            if self.performance_mode == 'fast':
                unique_faces = all_faces  # Skip duplicate removal for speed
            else:
                unique_faces = self._remove_duplicate_faces_fast(all_faces)
            
            person_height = y2 - y1
            
            valid_faces = []
            for (fx, fy, fw, fh, face_type) in unique_faces:
                # Check if face size is reasonable
                face_ratio = fh / person_height
                
                if self.min_face_ratio <= face_ratio <= self.max_face_ratio:
                    # Check face position (more lenient)
                    face_y_ratio = fy / person_height
                    
                    if face_y_ratio <= 0.7:  # Face in upper 70% of person
                        # Simplified quality assessment
                        if self.performance_mode == 'fast':
                            face_quality = 0.5  # Lower quality requirement for speed
                        else:
                            face_roi = gray_roi[fy:fy+fh, fx:fx+fw]
                            face_quality = self._assess_face_quality_fast(face_roi)
                        
                        if face_quality > confidence_threshold:
                            valid_faces.append({
                                'bbox': [int(fx + x1), int(fy + y1), int(fw), int(fh)],
                                'quality': float(face_quality),
                                'face_ratio': float(face_ratio),
                                'position_ratio': float(face_y_ratio),
                                'type': str(face_type)
                            })
            
            has_visible_face = len(valid_faces) > 0
            
            face_info = {
                'visible_faces': valid_faces,
                'total_faces_detected': int(len(unique_faces)),
                'best_face_quality': float(max([f['quality'] for f in valid_faces]) if valid_faces else 0),
                'person_size': (int(x2-x1), int(person_height)),
                'face_types': [str(f['type']) for f in valid_faces]
            }
            
            return has_visible_face, face_info
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return False, {}
    
    def _remove_duplicate_faces_fast(self, all_faces, overlap_threshold=0.4):
        """
        Fast duplicate removal with higher overlap threshold
        """
        if not all_faces:
            return []
        
        # Sort by area (larger faces first)
        all_faces.sort(key=lambda x: x[2] * x[3], reverse=True)
        
        unique_faces = []
        for face in all_faces:
            x1, y1, w1, h1, face_type = face
            is_duplicate = False
            
            # Only check against first few existing faces for performance
            for existing_face in unique_faces[:3]:  # Limit comparison
                x2, y2, w2, h2, _ = existing_face
                
                # Quick overlap check
                if (abs(x1 - x2) < max(w1, w2) * 0.5 and 
                    abs(y1 - y2) < max(h1, h2) * 0.5):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _assess_face_quality_fast(self, face_roi):
        """
        Fast face quality assessment
        """
        if face_roi.size == 0:
            return 0.0
        
        try:
            # Quick quality metrics
            contrast = face_roi.std() / 255.0
            mean_brightness = np.mean(face_roi) / 255.0
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            
            # Size check
            h, w = face_roi.shape
            size_score = 1.0 if min(h, w) >= 15 else min(h, w) / 15.0
            
            # Simple combination
            quality_score = (contrast * 0.4 + brightness_score * 0.3 + size_score * 0.3)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception:
            return 0.0
    
    def annotate_frontal_persons(self, frame, person_detections, frontal_results):
        """
        Lightweight annotation
        """
        annotated_frame = frame.copy()
        
        for i, (person_bbox, (has_visible_face, face_info)) in enumerate(zip(person_detections, frontal_results)):
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox['bbox']]
            
            # Color coding
            color = (0, 255, 0) if has_visible_face else (0, 0, 255)
            thickness = 3 if has_visible_face else 2
            
            # Draw person bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Simple label
            label = f"Face" if has_visible_face else "Person"
            if 'confidence' in person_bbox:
                label += f" {person_bbox['confidence']:.2f}"
                
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw faces with demographics (simplified)
            if has_visible_face and face_info.get('visible_faces') and self.performance_mode != 'fast':
                for face in face_info['visible_faces'][:2]:  # Limit to 2 faces for performance
                    fx, fy, fw, fh = face['bbox']
                    cv2.rectangle(annotated_frame, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 1)
                    
                    # Add demographics info if available
                    if 'demographics' in face:
                        demo = face['demographics']
                        gender = demo.get('gender', 'Unknown')
                        age = demo.get('age_group', 'Unknown')
                        demo_text = f"{gender[:1]},{age}"
                        cv2.putText(annotated_frame, demo_text, (fx, fy-25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return annotated_frame
    
    def should_process_frame(self):
        """
        Frame skip logic for performance optimization
        """
        self.frame_counter += 1
        return self.frame_counter % self.frame_skip == 0
    
    def filter_frontal_persons(self, frame, yolo_results):
        """
        Optimized person filtering with frame skipping
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
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence)
                }
                all_persons.append(person_data)
                
                # Face detection with frame skipping
                if self.should_process_frame():
                    is_visible, face_info = self.is_person_frontal(frame, [x1, y1, x2, y2])
                    
                    # Add demographics analysis if faces are visible
                    if is_visible and face_info.get('visible_faces'):
                        demographics = gender_age_detector.analyze_person_demographics(
                            frame, [x1, y1, x2, y2], face_info['visible_faces']
                        )
                        
                        # Add demographics to each face
                        for face in face_info['visible_faces']:
                            face['demographics'] = demographics
                        
                        # Add demographics to person data
                        person_data['demographics'] = demographics
                    
                    frontal_results.append((is_visible, face_info))
                    
                    if is_visible:
                        person_data['face_info'] = face_info
                        frontal_persons.append(person_data)
                    
                    # Store results for skipped frames
                    self.last_detection_results = frontal_results
                else:
                    # Use last detection results for skipped frames
                    if len(self.last_detection_results) > len(frontal_results):
                        idx = len(frontal_results)
                        if idx < len(self.last_detection_results):
                            is_visible, face_info = self.last_detection_results[idx]
                            frontal_results.append((is_visible, face_info))
                            
                            if is_visible:
                                person_data['face_info'] = face_info
                                frontal_persons.append(person_data)
                    else:
                        # Default to no face detected for new persons
                        frontal_results.append((False, {}))
        
        return frontal_persons, all_persons, frontal_results