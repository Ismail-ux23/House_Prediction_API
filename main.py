import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import io





app = FastAPI()
frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

model = joblib.load("house_price_model.joblib")
model_columns = joblib.load("house_price_model_columns.joblib")
average_error = 32773

class HouseData(BaseModel):
    MedInc: float = Field(..., description="Median income")
    HouseAge: float = Field(..., description="House age")
    AveRooms: float = Field(..., description="Average rooms")
    AveBedrms: float = Field(..., description="Average bedrooms")
    Population: float = Field(..., description="Population")
    AveOccup: float = Field(..., description="Average occupancy")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")

@app.get("/")
def home():
    return FileResponse(frontend_dir / "index.html")


@app.get("/guide")
def guide():
    return FileResponse(frontend_dir / "guide.html")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model":"RandomForestRegressor",
        "features": model_columns,
        "avg_error": f"${average_error:,.0f}"
        }

@app.post("/predict")
def predict(data: HouseData):
    try:
        input_data =pd.DataFrame([{
            "MedInc": data.MedInc,
            "HouseAge": data.HouseAge,
            "AveRooms": data.AveRooms,
            "AveBedrms": data.AveBedrms,
            "Population": data.Population,
            "AveOccup": data.AveOccup,
            "Latitude": data.Latitude,
            "Longitude": data.Longitude
        }])
        prediction = float(model.predict(input_data)[0])
        price = prediction * 100000
        return {
            "predicted_price": f"${price:,.0f}",
            "predicted_price_short": f"${prediction:,.2f} hundred thousand",
            "fidence_range": f"${price - average_error:,.0f} - ${price + average_error:,.0f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed: {str(e)}")


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read the CSV file: {e}")

    required_columns = set(model_columns)
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"Missing required columns. Required columns are: {', '.join(required_columns)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.") 

    try:
        predictions = model.predict(df[model_columns])
        df["predicted_price"] = predictions * 100000
        df["predicted_price_short"] = predictions
        df["confidence_range"] = df["predicted_price"].apply(lambda x: f"${x - average_error:,.0f} - ${x + average_error:,.0f}")

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")