from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "Running"


def test_predict_customer():

    response = client.post(
        "/predict",
        json={
            "annual_income": 53,
            "spending_score": 23
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_cluster" in data

    assert 1 <= data["predicted_cluster"] <= 5


def test_invalid_customer():

    response = client.post(
        "/predict",
        json={
            "annual_income": 53
        }
    )

    assert response.status_code == 422


def test_centroids():

    response = client.get("/centroids")

    assert response.status_code == 200

    assert "centroids" in response.json()


def test_batch_prediction():

    response = client.post(
        "/predict-batch",
        json=[
            {
                "annual_income": 53,
                "spending_score": 23
            },
            {
                "annual_income": 85,
                "spending_score": 90
            }
        ]
    )

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == 2

    assert all(1 <= cluster <= 5 for cluster in predictions)