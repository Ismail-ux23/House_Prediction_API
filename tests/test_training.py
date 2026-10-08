"""Exercise the real training/serialization code without downloading a dataset."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import main


def test_training_saves_loadable_model_and_measured_error(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.uniform(1, 10, (40, 8)), columns=main.FEATURES)
    target = features['MedInc'] * 0.2
    dataset = SimpleNamespace(data=features, target=target, feature_names=main.FEATURES)
    monkeypatch.setattr('sklearn.datasets.fetch_california_housing', lambda **kwargs: dataset)
    script = tmp_path / 'train.py'
    script.write_text((main.BASE_DIR / 'train.py').read_text())
    monkeypatch.chdir(tmp_path.parent)
    runpy.run_path(str(script))
    model = joblib.load(tmp_path / 'house_price_model.joblib')
    metadata = json.loads((tmp_path / 'house_price_model_metadata.json').read_text())
    _, x_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    assert metadata['mae_dollars'] == mean_absolute_error(y_test, model.predict(x_test)) * 100000
    assert metadata['features'] == main.FEATURES
    assert joblib.load(tmp_path / 'house_price_model_columns.joblib') == main.FEATURES
