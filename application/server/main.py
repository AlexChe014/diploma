import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.routing import Route, Mount
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



app = FastAPI(title='Pneumonia Prediction')
templates = Jinja2Templates(directory="application/templates/")

model = load_model(os.path.join(os.getcwd(), 'application', 'model_2.h5'))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
app.mount("/static", StaticFiles(directory="application/static/"), name="static")


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
    image = plt.imread("application/images/" + file.filename)

    image_new = cv.resize(image,(IMG_SIZE, IMG_SIZE))
    image_new = image_new.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    result = model.predict(image_new)
    if(int(result[0][0]) == 0):
        result = "0"
    else:
        result = "1"

    return {"path":os.path.exists("application/images/" + file.filename), "file":file.filename, "class": result}





if __name__ == "__main__":
    uvicorn.run(app, debug=True)
