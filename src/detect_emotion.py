import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("📸 Starting Emotion Detection...")

# Load your trained model
model = load_model('model/emotion_model.keras')
print("✅ Model loaded!")

# Define emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Use correct camera index (adjust if needed)
cap = cv2.VideoCapture(1)  # 1, 2, or 3 for DroidCam USB

# Check if camera is opened
if not cap.isOpened():
    print("❌ Error: Could not open video stream.")
    exit()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("✅ Cascade loaded. Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)
        emotion = labels[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2)

    # Show the frame
    cv2.imshow("📱 Emotion Detection (Phone Cam)", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()