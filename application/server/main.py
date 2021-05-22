import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.responses import RedirectResponse
import os
import cv2 as cv
from io import BytesIO
import aiofiles
import matplotlib.pyplot as plt
import numpy as np
from application.components import predict, read_imagefile
from application.schema import Symptom
from PIL import Image
from tensorflow.keras.models import load_model



app = FastAPI(title='Tensorflow FastAPI Starter Pack')
model = load_model(os.path.join(os.getcwd(), 'application', 'model_2.h5'))

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    file_location = os.path.join(os.getcwd(), 'application', "images", file.filename)
    async with aiofiles.open(file_location, 'wb+') as out_file:
        content = await file.read()
        await out_file.write(content) 
    IMG_SIZE = 150
    image = cv.imread("application/images/" + file.filename, cv.IMREAD_GRAYSCALE)
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_new = cv.resize(image,(IMG_SIZE, IMG_SIZE))
    image_new = image_new.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    #image = read_imagefile(file_location)
    result = model.predict_classes(image_new)

    return {"class": result[0][0]}





if __name__ == "__main__":
    uvicorn.run(app, debug=True)
