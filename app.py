
import os
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

import asyncio


disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1.pt"))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)


@app.route('/disease', methods=['POST', 'GET'])
def disease():
    if request.method == "POST":
        image = request.files['image']
        filename = image.filename
        print(image, filename)
        filepath = os.path.join('uploads', filename)
        image.save(filepath)
        pred = prediction(filepath)
        print(pred)
        title = disease_info['title'][pred]
        status = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        return jsonify(
            {
                "message": "Successfully uploaded"
            }
        )
    else:
        return "HelloweeeeenS"


app.run(debug=True, host="192.168.1.24", port=4000)
