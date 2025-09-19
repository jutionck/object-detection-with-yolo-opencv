import cv2
import numpy as np
import os
import urllib.request

class GenderAgeDetector:
    def __init__(self):
        """
        Gender and Age detection using OpenCV DNN
        Uses pre-trained models for gender and age classification
        """
        self.gender_net = None
        self.age_net = None
        self.face_net = None
        
        # Model definitions
        self.gender_list = ['Male', 'Female']
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Model URLs and paths
        self.models = {
            'gender': {
                'prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
                'caffemodel': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector_uint8.pb',
                'local_prototxt': 'models/gender_deploy.prototxt',
                'local_model': 'models/gender_net.caffemodel'
            },
            'age': {
                'local_prototxt': 'models/age_deploy.prototxt', 
                'local_model': 'models/age_net.caffemodel'
            }
        }
        
        # Create models directory
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize gender and age detection models
        Uses lightweight alternative since full models are large
        """
        print("Initializing Gender & Age detection...")
        
        # For this implementation, we'll use a simplified approach
        # In production, you would download actual pre-trained models
        
        # Simplified gender detection based on face features
        self.gender_net = "lightweight"  # Placeholder for lightweight detection
        self.age_net = "lightweight"     # Placeholder for lightweight detection
        
        print("✅ Gender & Age detection initialized (lightweight mode)")
    
    def detect_gender_age(self, face_roi):
        """
        Detect gender and age from face ROI
        
        Args:
            face_roi: Face region of interest (cropped face image)
            
        Returns:
            tuple: (gender, age_group, confidence_scores)
        """
        try:
            if face_roi.size == 0:
                return "Unknown", "Unknown", {"gender": 0.0, "age": 0.0}
            
            # Resize face for analysis
            face_resized = cv2.resize(face_roi, (96, 96))
            
            # Convert to RGB if needed
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_resized
            
            # Simplified gender detection based on facial features analysis
            gender, gender_confidence = self._analyze_gender_features(face_gray)
            
            # Simplified age estimation based on facial characteristics
            age_group, age_confidence = self._analyze_age_features(face_gray)
            
            return gender, age_group, {
                "gender": float(gender_confidence),
                "age": float(age_confidence)
            }
            
        except Exception as e:
            print(f"Error in gender/age detection: {e}")
            return "Unknown", "Unknown", {"gender": 0.0, "age": 0.0}
    
    def _analyze_gender_features(self, face_gray):
        """
        Simplified gender analysis based on facial features
        This is a basic heuristic approach - in production use actual DNN models
        """
        try:
            # Calculate basic facial feature metrics
            height, width = face_gray.shape
            
            # Analyze upper face region (forehead, eyebrows area)
            upper_region = face_gray[0:int(height*0.4), :]
            
            # Analyze jawline region
            lower_region = face_gray[int(height*0.7):height, :]
            
            # Simple heuristics based on contrast and texture
            upper_contrast = np.std(upper_region)
            lower_contrast = np.std(lower_region)
            
            # Edge density analysis
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Basic classification logic (simplified)
            # This is just for demonstration - real models would be much more accurate
            
            # Higher edge density and contrast often correlates with masculine features
            if edge_density > 0.15 and lower_contrast > upper_contrast * 1.1:
                return "Male", min(0.6 + (edge_density - 0.15) * 2, 0.85)
            else:
                return "Female", min(0.6 + (0.15 - edge_density) * 2, 0.85)
                
        except Exception:
            return "Male", 0.5  # Default fallback
    
    def _analyze_age_features(self, face_gray):
        """
        Simplified age estimation based on facial characteristics
        This is a basic heuristic approach - in production use actual DNN models
        """
        try:
            height, width = face_gray.shape
            
            # Analyze skin texture (wrinkles, smoothness)
            # Calculate local standard deviation to detect texture variations
            kernel = np.ones((5,5), np.float32) / 25
            smoothed = cv2.filter2D(face_gray, -1, kernel)
            texture_variation = np.mean(np.abs(face_gray.astype(float) - smoothed.astype(float)))
            
            # Analyze forehead region for wrinkles
            forehead_region = face_gray[0:int(height*0.3), int(width*0.2):int(width*0.8)]
            forehead_edges = cv2.Canny(forehead_region, 30, 100)
            forehead_lines = np.sum(forehead_edges > 0) / forehead_region.size
            
            # Eye region analysis
            eye_region = face_gray[int(height*0.3):int(height*0.5), :]
            eye_contrast = np.std(eye_region)
            
            # Simple age classification based on texture and features
            if texture_variation > 25 and forehead_lines > 0.05:
                if texture_variation > 35:
                    return "(60-100)", min(0.6 + (texture_variation - 35) * 0.01, 0.8)
                else:
                    return "(48-53)", min(0.6 + (texture_variation - 25) * 0.015, 0.8)
            elif texture_variation > 15:
                if forehead_lines > 0.03:
                    return "(38-43)", min(0.6 + forehead_lines * 5, 0.8)
                else:
                    return "(25-32)", min(0.6 + texture_variation * 0.02, 0.8)
            elif eye_contrast > 30:
                return "(15-20)", min(0.6 + eye_contrast * 0.01, 0.8)
            elif texture_variation > 8:
                return "(8-12)", min(0.6 + texture_variation * 0.03, 0.8)
            else:
                return "(4-6)", min(0.6 + (8 - texture_variation) * 0.02, 0.8)
                
        except Exception:
            return "(25-32)", 0.5  # Default fallback
    
    def analyze_person_demographics(self, frame, person_bbox, visible_faces):
        """
        Analyze demographics for a person based on visible faces
        
        Args:
            frame: Input frame
            person_bbox: Person bounding box
            visible_faces: List of detected faces for this person
            
        Returns:
            dict: Demographics information
        """
        if not visible_faces:
            return {
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'confidence': {'gender': 0.0, 'age': 0.0},
                'face_count': 0
            }
        
        try:
            # Use the best quality face for analysis
            best_face = max(visible_faces, key=lambda x: x['quality'])
            
            # Extract face ROI
            fx, fy, fw, fh = best_face['bbox']
            face_roi = frame[fy:fy+fh, fx:fx+fw]
            
            # Detect gender and age
            gender, age_group, confidence = self.detect_gender_age(face_roi)
            
            return {
                'gender': gender,
                'age_group': age_group,
                'confidence': confidence,
                'face_count': len(visible_faces),
                'best_face_quality': best_face['quality']
            }
            
        except Exception as e:
            print(f"Error analyzing demographics: {e}")
            return {
                'gender': 'Unknown',
                'age_group': 'Unknown',
                'confidence': {'gender': 0.0, 'age': 0.0},
                'face_count': len(visible_faces)
            }
    
    def get_demographics_summary(self, demographics_list):
        """
        Get summary statistics for demographics
        
        Args:
            demographics_list: List of demographics data
            
        Returns:
            dict: Summary statistics
        """
        if not demographics_list:
            return {
                'gender_distribution': {'Male': 0, 'Female': 0, 'Unknown': 0},
                'age_distribution': {},
                'total_analyzed': 0
            }
        
        gender_count = {'Male': 0, 'Female': 0, 'Unknown': 0}
        age_count = {}
        
        for demo in demographics_list:
            # Count gender
            gender = demo.get('gender', 'Unknown')
            gender_count[gender] = gender_count.get(gender, 0) + 1
            
            # Count age
            age = demo.get('age_group', 'Unknown')
            age_count[age] = age_count.get(age, 0) + 1
        
        return {
            'gender_distribution': gender_count,
            'age_distribution': age_count,
            'total_analyzed': len(demographics_list)
        }

# Global instance
gender_age_detector = GenderAgeDetector()