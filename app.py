import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from textblob import TextBlob

# Initialize Flask App
app = Flask(__name__)

# ------------------------- MODEL LOADING -------------------------

def load_sign_model():
    model_dir = "model"
    h5_path = os.path.join(model_dir, "model-bw_dru.h5")  # 👈 your actual file name

    try:
        if os.path.exists(h5_path):
            print("📂 Loading model directly from .h5 file...")
            model = load_model(h5_path, compile=False)
            print("✅ Model loaded successfully from .h5 file.")
            return model
        else:
            print("❌ model-bw_dru.h5 not found in 'model' folder.")
            return None
    except Exception as e:
        print(f"⚠️ Failed to load model properly: {e}")
        return None

model = load_sign_model()

# ------------------------- CAMERA SETUP -------------------------

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("⚠️ Warning: Camera not accessible.")

# ------------------------- FRAME GENERATOR -------------------------

def generate_frames():
    """
    Continuously capture frames from webcam,
    detect ROI and run sign language model prediction.
    """
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        roi = frame[100:400, 320:620]
        cv2.rectangle(frame, (320, 100), (620, 400), (0, 255, 0), 2)

        if model is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))
            gray = gray.reshape(1, 128, 128, 1) / 255.0

            # Model prediction
            prediction = model.predict(gray)
            predicted_class = np.argmax(prediction)
            text = f"Predicted: {predicted_class}"
            cv2.putText(frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Model not loaded!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------------- GRAMMAR CORRECTION -------------------------

@app.route("/correct/<text>")
def correct_text(text):
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return jsonify({"original": text, "corrected": corrected})
    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------------- ROUTES -------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------- MAIN -------------------------

if __name__ == "__main__":
    print("🚀 Starting Application...")
    app.run(debug=True)
