# House Prediction API

FastAPI service and browser interface for single-row and CSV batch estimates
using scikit-learn's California Housing dataset and a Random Forest regressor.
These are historical California district median house values in USD, not current
property appraisals or estimates for Pakistan.

## Run locally

Use Python 3.12, create and activate a virtual environment, then:

```bash
python -m pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload
```

Training downloads the California Housing dataset on its first run, uses an
80/20 split with seed 42, and saves the model, ordered feature columns and held-out
MAE metadata beside the scripts. Model binaries and generated metadata are
ignored by git. Train and serve using the same installed scikit-learn version.
Only load trusted local joblib artifacts; loading them can execute Python code.

Open http://127.0.0.1:8000 for the UI, `/guide` for guidance, or `/docs` for the
API schema. Without model artifacts the pages still load, but `/health` and
prediction requests return 503 with setup guidance. File paths resolve relative
to the scripts rather than the terminal's current directory.

## Prediction inputs

Both routes use these eight features in the same order:

| Feature | Meaning | Accepted range |
| --- | --- | --- |
| MedInc | District median income in $10,000 units | Positive |
| HouseAge | Median house age in years | Nonnegative |
| AveRooms | Average rooms per household | Positive |
| AveBedrms | Average bedrooms per household | Positive |
| Population | District population | Positive |
| AveOccup | Average household occupancy | Positive |
| Latitude | Latitude in degrees | -90 to 90 |
| Longitude | Longitude in degrees | -180 to 180 |

All values must be finite numbers. Bounds reject malformed data; they do not
make predictions outside the training distribution reliable.

`POST /predict` accepts a JSON object. Example:

```json
{"MedInc":8.3,"HouseAge":41,"AveRooms":7,"AveBedrms":1,"Population":1000,"AveOccup":3,"Latitude":34,"Longitude":-118}
```

`POST /predict_file` accepts multipart field `file`: a UTF-8 CSV with the eight
required headers. Extra columns are preserved. Duplicate headers, malformed
rows, missing values and invalid numbers return 400 with no partial predictions.
Uploads are limited to 5 MiB and 10,000 rows; excess returns 413. The response is
a downloadable `predictions.csv` with numeric `predicted_price` in USD,
`predicted_price_short` in $100,000 units, and `estimated_error_range` (plus deprecated `confidence_range` alias).

The displayed error band uses the MAE measured during training. **It is not a
confidence interval or a guaranteed price range.** JSON retains `fidence_range`
as a deprecated alias; the UI uses `estimated_error_range`.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node tests/test_csv.cjs
```

The small CSV preview test requires Node.js 18 or later.

Tests use a deterministic fake predictor so CI needs neither a model binary nor
a dataset download. They verify startup without artifacts, feature ordering,
USD conversion, input validation, CSV quoting, upload limits and error handling.
The training pipeline is also checked on a small synthetic dataset, including saved model loading and MAE calculation. These tests do not measure real California Housing accuracy. No deployed service is included.
