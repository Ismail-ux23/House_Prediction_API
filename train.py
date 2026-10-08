from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

data = fetch_california_housing(as_frame=True)
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print (f"total records: {x.shape[0]}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Average error: ${mean_absolute_error(y_test, y_pred)*100000:,.0f}")

joblib.dump(model, BASE_DIR / "house_price_model.joblib")
joblib.dump(list(x.columns), BASE_DIR / "house_price_model_columns.joblib")
(BASE_DIR / "house_price_model_metadata.json").write_text(json.dumps({"mae_dollars": float(mean_absolute_error(y_test, y_pred) * 100000), "features": list(x.columns), "random_state": 42}, indent=2))