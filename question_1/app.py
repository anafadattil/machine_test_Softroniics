from flask import Flask, Response
import cv2
from detect import FaceHandDetector

app = Flask(__name__)
detector = FaceHandDetector()

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = Default webcam
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Detect faces & hands
        processed_frame = detector.detect(frame)
        
        # Convert to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return "Welcome to Real-Time Face & Hand Detection!"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)