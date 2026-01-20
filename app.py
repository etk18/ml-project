from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Mount static files if you have any (uncomment if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Route for home page - landing page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    """GET request for prediction page"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    gender: str = Form(...),
    ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    writing_score: float = Form(...),
    reading_score: float = Form(...)
):
    """POST request for prediction"""
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    
    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("after Prediction")
    
    return templates.TemplateResponse(
        "home.html", 
        {"request": request, "results": results[0]}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
