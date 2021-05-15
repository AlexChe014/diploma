import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import os
from application.components import predict, read_imagefile
from application.schema import Symptom



app = FastAPI(title='Tensorflow FastAPI Starter Pack')


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    file_location = os.path.join(os.getcwd(), 'application', "images", file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    image = read_imagefile(file_location)
    prediction = predict(image)

    return prediction





if __name__ == "__main__":
    uvicorn.run(app, debug=True)
