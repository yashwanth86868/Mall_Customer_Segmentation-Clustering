from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Create FastAPI application
app = FastAPI(
    title="Mall Customer Segmentation API",
    description="K-Means Customer Segmentation REST API",
    version="1.0"
)

# Load trained ML model
with open("kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)


# Define input schema
class Customer(BaseModel):
    annual_income: float
    spending_score: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Mall Customer Segmentation API",
        "status": "Running"
    }


# Prediction endpoint
@app.post("/predict")




def predict_customer(customer: Customer):

    input_data = np.array([[
        customer.annual_income,
        customer.spending_score
    ]])

    prediction = model.predict(input_data)

    return {
        "annual_income": customer.annual_income,
        "spending_score": customer.spending_score,
        "predicted_cluster": int(prediction[0] + 1)
    }





# Get cluster centroids
@app.get("/centroids")
def get_centroids():

    return {
        "centroids": model.cluster_centers_.tolist()
    }


# Predict multiple customers
@app.post("/predict-batch")
def predict_batch(customers: list[Customer]):

    input_data = np.array([
        [c.annual_income, c.spending_score]
        for c in customers
    ])

    predictions = model.predict(input_data)

    return {
        "predictions": [
            int(cluster + 1)
            for cluster in predictions
        ]
    }
    
    
    
def predict_customer(customer: Customer):

    # Prepare input
    input_data = np.array([[
        customer.annual_income,
        customer.spending_score
    ]])

    # Predict cluster
    prediction = model.predict(input_data)

    return {
        "annual_income": customer.annual_income,
        "spending_score": customer.spending_score,
        "predicted_cluster": int(prediction[0] + 1)
    }