from io import BytesIO
import os
import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.models import load_model

model = load_model(os.path.join(os.getcwd(), 'application', 'model_2.h5'))




def predict(image: np.array):
    global model

    #image = np.asarray(image.resize((224, 224)))[..., :3]
    #image = np.expand_dims(image, 0)
    #image = image / 127.5 - 1.0

    result = model.predict(image)

    #response = []
    #for i, res in enumerate(result):
    #    resp = {}
    #    resp["class"] = res[1]
    #    resp["confidence"] = f"{res[2]*100:0.2f} %"

    #    response.append(resp)
    #response = model.predict_classes(image)
    return result


def read_imagefile(file):# -> Image.Image:
    IMG_SIZE = 150
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image,(IMG_SIZE, IMG_SIZE))
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    #image = Image.open(BytesIO(file))
    return image
