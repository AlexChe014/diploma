from io import BytesIO
import os
import numpy as np
import tensorflow as tf
import cv2 as cv
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'application', 'model.h5'))


def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Model loaded")
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    #image = np.asarray(image.resize((224, 224)))[..., :3]
    #image = np.expand_dims(image, 0)
    #image = image / 127.5 - 1.0

    #result = decode_predictions(model.predict(image), 2)[0]

    #response = []
    #for i, res in enumerate(result):
    #   resp = {}
    #    resp["class"] = res[1]
    #    resp["confidence"] = f"{res[2]*100:0.2f} %"

    #    response.append(resp)
    response = model.predict_classes(image)
    return response


def read_imagefile(file):# -> Image.Image:
    IMG_SIZE = 100
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image,(IMG_SIZE, IMG_SIZE))
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    #image = Image.open(BytesIO(file))
    return image
