from fastapi import FastAPI, File, HTTPException, UploadFile

from inference import load_model, predict_from_bytes

app = FastAPI(title="Medicinal Plant Classifier API")

# Load the model once when the server starts
MODEL_PATH = "model/medicinal_plant_classifier.pth"
model, class_names, device = load_model(MODEL_PATH)


@app.get("/")
def root():
    return {"status": "ok", "model": "medicinal_plant_classifier"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    image_bytes = await file.read()
    try:
        label, confidence = predict_from_bytes(model, class_names, device, image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"plant_name": label, "confidence": confidence}
