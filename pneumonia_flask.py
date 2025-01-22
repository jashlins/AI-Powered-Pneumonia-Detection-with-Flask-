from flask import Flask, request, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Path configurations
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "pneumonia_detector_final.keras"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your pre-trained model
model = load_model(MODEL_PATH)

# Define the preprocess and predict functions
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224 ))  # Ensure this matches the model input
    img_array = img_to_array(img) / 255.0  # Normalize the image
    return img_array[np.newaxis, ...]  # Add batch dimension

def predict(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    confidence = float(prediction[0][0])
    return label, confidence

# Define routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Save the uploaded file
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Get prediction
            label, confidence = predict(filepath)

            # Render result
            return render_template(
                "result.html",
                label=label,
                confidence=f"{confidence:.2f}",
                image_path=filepath,
            )
    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)
