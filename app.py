from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_aggregate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid Image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    if brightness > 200:
        return "Rejected: Overexposed"
    elif brightness < 40:
        return "Rejected: Too Dark"
    elif edge_density > 0.08:
        return "Accepted: Aggregate Detected"
    else:
        return "Rejected: Not Aggregate"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = detect_aggregate(path)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)