from flask import Flask, request,jsonify,send_file
import cv2
import numpy as numpy
import torch
import os
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('Pred.pt')

up_folder = 'uploads'
result_folder = 'results'

os.makedirs(up_folder,exist_ok=True)
os.makedirs(result_folder,exist_ok=True)

@app.route('/')
def home():
    return "Weed Detection API"


@app.route('/predict',methods=["POST"])
def pred():
    if 'image' not in request.files:
        return jsonify({'error':'No Image Provided'}),400

    file = request.files['image']
    fname = os.path.join(up_folder,file.filename)
    file.save(fname)

    image = cv2.imread(fname)

    results = model.predict(image,save=True,conf=0.5)

    result_image_path = os.path.join(result_folder, file.filename)
    results[0].save(result_image_path)

    return send_file(result_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)  