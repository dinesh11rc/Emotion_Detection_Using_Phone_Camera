import cv2

print("🔍 Checking camera indexes...")

for i in range(5):
    print(f"Trying index {i}...")
    cap = cv2.VideoCapture(1)
    if cap is None or not cap.isOpened():
        print(f"❌ Index {i} not working")
    else:
        print(f"✅ Index {i} is working!")
        cap.release()