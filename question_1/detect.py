import cv2
import mediapipe as mp

class FaceHandDetector:
    def __init__(self):
        # Initialize MediaPipe models
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.5)
        self.hand_detector = self.mp_hands.Hands(min_detection_confidence=0.5)
        
        # Drawing utilities (for landmarks)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and hands
        face_results = self.face_detector.process(rgb_frame)
        hand_results = self.hand_detector.process(rgb_frame)
        
        # Draw face detections
        if face_results.detections:
            for detection in face_results.detections:
                self.mp_draw.draw_detection(frame, detection)
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame