import cv2
from model_loader import load_model
from utils import preprocess_image

class MoodDetector:
    def __init__(self):
        self.model = load_model()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_mood(self):
        """Real-time mood detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                processed_face = preprocess_image(face)
                
                # Predict emotion
                prediction = self.model.predict(processed_face)
                emotion = self.emotion_labels[np.argmax(prediction)]
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                
                # Special handling for "crying" (detect as 'Sad' with low confidence)
                if emotion == 'Sad' and prediction[0][4] > 0.7:
                    cv2.putText(frame, "Crying detected!", (x, y+h+30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.imshow('Mood Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MoodDetector()
    detector.detect_mood()
