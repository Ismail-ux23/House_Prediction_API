"""California housing estimates; run train.py to create local model artifacts."""
import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, ValidationError

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 10_000
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.average_error = None
    try:
        # Only load locally trained/trusted joblib files: they can execute code.
        app.state.model = joblib.load(BASE_DIR / 'house_price_model.joblib')
        metadata = json.loads((BASE_DIR / 'house_price_model_metadata.json').read_text())
        if metadata['features'] != FEATURES:
            raise ValueError('Feature schema mismatch')
        error = float(metadata['mae_dollars'])
        if not np.isfinite(error) or error < 0:
            raise ValueError('Invalid MAE')
        app.state.average_error = error
    except Exception:
        app.state.model = None
        logger.warning('Model unavailable. Run python train.py to generate model and metadata.')
    yield


app = FastAPI(lifespan=lifespan)
frontend_dir = BASE_DIR / 'frontend'
app.mount('/static', StaticFiles(directory=frontend_dir), name='static')


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Raw invalid inputs can contain NaN/Infinity, which JSON cannot serialize.
    errors = [{key: error[key] for key in ('loc', 'msg', 'type')}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': errors})


class HouseData(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


def require_model():
    if getattr(app.state, 'model', None) is None:
        raise HTTPException(503, 'Model unavailable. Run python train.py before predicting.')
    return app.state.model


def predict_prices(frame):
    model = require_model()
    try:
        values = np.asarray(model.predict(frame[FEATURES]), dtype=float)
        prices = values * 100_000
        if prices.shape != (len(frame),) or not np.isfinite(prices).all() or (prices < 0).any():
            raise ValueError('Invalid model output')
        return prices
    except Exception:
        logger.exception('Prediction failed')
        raise HTTPException(500, 'Prediction failed. Check the server logs.') from None


def error_range(price):
    # Held-out MAE is a typical error magnitude, not a confidence interval.
    error = getattr(app.state, 'average_error', None)
    return None if error is None else f'${max(0, price - error):,.0f} - ${price + error:,.0f}'


@app.get('/')
def home():
    return FileResponse(frontend_dir / 'index.html')


@app.get('/guide')
def guide():
    return FileResponse(frontend_dir / 'guide.html')


@app.get('/health')
def health_check():
    available = getattr(app.state, 'model', None) is not None
    return JSONResponse(status_code=200 if available else 503, content={
        'status': 'healthy' if available else 'model_unavailable',
        'features': FEATURES, 'mae_dollars': getattr(app.state, 'average_error', None),
    })


@app.post('/predict')
def predict(data: HouseData):
    price = float(predict_prices(pd.DataFrame([data.model_dump()]))[0])
    estimate_range = error_range(price)
    return {
        'predicted_price': f'${price:,.0f}',
        'predicted_price_short': f'${price / 100_000:,.2f} hundred thousand',
        'estimated_error_range': estimate_range,
        # Retain the misspelled field for existing API clients.
        'fidence_range': estimate_range,
    }


@app.post('/predict_file')
async def predict_file(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise HTTPException(400, 'Please upload a CSV file.')
        content = await file.read(MAX_BYTES + 1)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, 'CSV files must be no larger than 5 MiB.')
        try:
            text = content.decode('utf-8-sig')
            reader = csv.reader(io.StringIO(text), strict=True)
            headers = next(reader)
            if len(headers) != len(set(headers)):
                raise HTTPException(400, 'CSV column names must be unique.')
            if not set(FEATURES).issubset(headers):
                raise HTTPException(400, 'Missing required columns: ' + ', '.join(FEATURES))
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(headers):
                    raise HTTPException(400, 'Each CSV row must match the number of header columns.')
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    raise HTTPException(413, 'CSV files must contain at most 10,000 data rows.')
            if not rows:
                raise HTTPException(400, 'CSV must contain at least one data row.')
        except (UnicodeDecodeError, csv.Error, StopIteration):
            raise HTTPException(400, 'Could not read the CSV. Use UTF-8 with a header row.') from None
        df = pd.DataFrame(rows, columns=headers)
        validated = []
        for index, row in df.iterrows():
            try:
                validated.append(HouseData(**{key: row[key] for key in FEATURES}).model_dump())
            except ValidationError:
                raise HTTPException(400, f'Invalid feature values in data row {index + 1}. Use finite numbers within the allowed ranges.') from None
        prices = predict_prices(pd.DataFrame(validated))
        df['predicted_price'] = prices
        df['predicted_price_short'] = prices / 100_000
        df['estimated_error_range'] = [error_range(float(price)) for price in prices]
        df['confidence_range'] = df['estimated_error_range']  # Deprecated CSV alias.
        output = io.StringIO()
        df.to_csv(output, index=False)
        return StreamingResponse(iter([output.getvalue()]), media_type='text/csv',
                                 headers={'Content-Disposition': 'attachment; filename=predictions.csv'})
    finally:
        await file.close()
