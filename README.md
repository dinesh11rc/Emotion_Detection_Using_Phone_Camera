# 📸 Emotion Detection via Phone Camera (DroidCam + MobileNetV2)

## 📖 Overview
This project is a **real-time emotion detection system** using **Deep Learning (MobileNetV2), OpenCV, and Flask** with phone camera input (via DroidCam).  
It classifies human emotions such as **Happy, Sad, Angry, Neutral, and Surprise** by analyzing live video feed.  

The project demonstrates how **AI + Computer Vision** can be applied to mental health, human-computer interaction, and smart applications.

---

## ✨ Features
- 🎥 Real-time emotion detection using phone camera  
- 🤖 Deep Learning with **MobileNetV2 + TensorFlow/Keras**  
- 🌐 Flask integration for running in a local web app  
- ⚡ Lightweight and efficient — works on CPU  
- 📊 Extendable for more emotion categories  

---

## 🛠 Requirements
- Python 3.10  
- TensorFlow 2.12  
- OpenCV  
- Flask  
- DroidCam (Android/iOS App + Windows Client)  

Install dependencies:
```
pip install -r requirements.txt
```
📂 Folder Structure
Emotion_Detection_Using_Phone_Camera/
│── model/                  # Trained model files (.keras)
│── src/                    # Source code
│   ├── detect_emotion.py   # Run emotion detection
│   ├── train_model.py      # Train the model
│   ├── check_camera_index.py # Verify camera source
│── requirements.txt        # Dependencies
│── README.md               # Documentation

1️⃣ Clone the Repository
git clone https://github.com/dinesh11rc/Emotion_Detection_Using_Phone_Camera.git
cd Emotion_Detection_Using_Phone_Camera

2️⃣ Setup Virtual Environment
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Emotion Detection
python src/detect_emotion.py

🔮 Future Scope

🌍 Deploy the model on cloud for remote access

📱 Mobile app integration (Android/iOS)

🧠 Enhance emotion categories (Fear, Disgust, Surprise, etc.)

💬 Real-time feedback and chatbot integration

📚 References

TensorFlow Documentation – https://www.tensorflow.org/

OpenCV Docs – https://docs.opencv.org/

Flask Docs – https://flask.palletsprojects.com/

DroidCam – https://www.dev47apps.com/

📝 License

This project is licensed under the MIT License – feel free to use and modify with credit.

