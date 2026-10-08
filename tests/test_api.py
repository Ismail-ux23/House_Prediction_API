import csv
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main

VALUES = dict(MedInc=8.3, HouseAge=41, AveRooms=7, AveBedrms=1, Population=1000, AveOccup=3, Latitude=34, Longitude=-118)


class FakeModel:
    def predict(self, frame):
        assert list(frame.columns) == main.FEATURES
        return np.full(len(frame), 2.5)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.joblib, 'load', lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    with TestClient(main.app) as client:
        main.app.state.model = FakeModel()
        main.app.state.average_error = 20_000
        yield client


def upload(client, rows=None, headers=None, content=None):
    if content is None:
        out = io.StringIO()
        writer = csv.writer(out)
        headers = headers or main.FEATURES
        writer.writerow(headers)
        for row in rows if rows is not None else [VALUES]:
            writer.writerow([row.get(key, '') for key in headers])
        content = out.getvalue().encode()
    return client.post('/predict_file', files={'file': ('houses.csv', content, 'text/csv')})


def test_missing_model_keeps_site_available(client):
    main.app.state.model = None
    assert client.get('/').status_code == 200
    assert client.get('/guide').status_code == 200
    assert client.get('/health').status_code == 503
    assert client.post('/predict', json=VALUES).status_code == 503
    assert upload(client).status_code == 503


def test_single_and_batch_use_same_units_and_feature_order(client):
    response = client.post('/predict', json=VALUES)
    assert response.status_code == 200
    assert response.json()['predicted_price'] == '$250,000'
    assert response.json()['estimated_error_range'] == '$230,000 - $270,000'
    rows = [dict(VALUES, note='comma, "quote"\nand new line'), dict(VALUES, note='second')]
    response = upload(client, rows=rows, headers=['note'] + list(reversed(main.FEATURES)))
    assert response.status_code == 200
    parsed = list(csv.DictReader(io.StringIO(response.text)))
    assert len(parsed) == 2
    assert parsed[0]['note'] == rows[0]['note']
    assert float(parsed[0]['predicted_price']) == 250_000


@pytest.mark.parametrize('key,value', [('MedInc', 0), ('HouseAge', -1), ('AveRooms', -1),
    ('AveBedrms', 0), ('Population', 0), ('AveOccup', 0), ('Latitude', 91), ('Longitude', -181),
    ('MedInc', 'nan'), ('MedInc', 'inf'), ('MedInc', 'abc'), ('MedInc', '')])
def test_single_and_csv_reject_invalid_values(client, key, value):
    data = dict(VALUES, **{key: value})
    assert client.post('/predict', json=data).status_code == 422
    assert upload(client, rows=[VALUES, data]).status_code == 400


@pytest.mark.parametrize('content', [b'', b'\xff\xfe', b'a,b\n1,2\n',
    (','.join(main.FEATURES) + '\n').encode(),
    (','.join(main.FEATURES + ['MedInc']) + '\n').encode(),
    (','.join(main.FEATURES) + '\n1,2\n').encode(),
    (','.join(main.FEATURES) + '\n"unterminated').encode()])
def test_malformed_csv_is_client_error(client, content):
    assert upload(client, content=content).status_code == 400


def test_upload_limits_and_extension(client, monkeypatch):
    assert client.post('/predict_file', files={'file': ('data.txt', b'x')}).status_code == 400
    monkeypatch.setattr(main, 'MAX_BYTES', 4)
    assert upload(client, content=b'12345').status_code == 413
    monkeypatch.setattr(main, 'MAX_BYTES', 10000)
    monkeypatch.setattr(main, 'MAX_ROWS', 1)
    assert upload(client, rows=[VALUES, VALUES]).status_code == 413


def test_bad_model_output_has_generic_error(client):
    class BadModel:
        def predict(self, frame):
            raise RuntimeError('private-server-file-path')
    main.app.state.model = BadModel()
    response = client.post('/predict', json=VALUES)
    assert response.status_code == 500
    assert 'private-server-file-path' not in response.text
    assert upload(client).status_code == 500


def test_utf8_bom_supported(client):
    content = (','.join(main.FEATURES) + '\n' + ','.join(str(VALUES[key]) for key in main.FEATURES)).encode('utf-8-sig')
    assert upload(client, content=content).status_code == 200


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_json_returns_serializable_validation_error(client, value):
    import json
    body = json.dumps(VALUES).replace('8.3', value)
    assert client.post('/predict', content=body, headers={'content-type': 'application/json'}).status_code == 422


def test_startup_loads_artifacts_from_script_directory(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(main, 'BASE_DIR', tmp_path)
    metadata = tmp_path / 'house_price_model_metadata.json'
    metadata.write_text(json.dumps({'mae_dollars': 12345, 'features': main.FEATURES}))
    paths = []
    def load(path):
        paths.append(path)
        return FakeModel()
    monkeypatch.setattr(main.joblib, 'load', load)
    monkeypatch.chdir(tmp_path.parent)
    with TestClient(main.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/health').json()['mae_dollars'] == 12345
        assert client.post('/predict', json=VALUES).status_code == 200
    assert paths == [tmp_path / 'house_price_model.joblib']


def test_corrupt_model_does_not_stop_site(monkeypatch):
    monkeypatch.setattr(main.joblib, 'load', lambda path: (_ for _ in ()).throw(EOFError()))
    with TestClient(main.app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/health').status_code == 503
