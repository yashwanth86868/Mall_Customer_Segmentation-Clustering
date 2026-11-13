
# 🛍️ Mall Customer Segmentation Using K-Means Clustering

## 📌 Project Overview

Mall Customer Segmentation is a Machine Learning project that groups customers into different segments based on their annual income and spending score.

This project uses the **K-Means Clustering algorithm**, an unsupervised machine learning technique, to identify customer groups and understand their purchasing behaviour.

The application provides interactive visualizations and customer cluster predictions through a Streamlit frontend connected to a FastAPI backend.

## 🎯 Project Objectives

- Identify different customer segments.
- Analyze customer purchasing behaviour.
- Group customers based on annual income and spending score.
- Visualize customer clusters using interactive charts.
- Provide cluster predictions through a REST API.

## 🛠️ Technologies Used

| Category             | Technologies                      |
| -------------------- | --------------------------------- |
| Programming Language | Python                            |
| Data Processing      | Pandas, NumPy                     |
| Machine Learning     | Scikit-learn, K-Means             |
| Data Visualization   | Matplotlib, Plotly                |
| Frontend             | Streamlit                         |
| Backend              | FastAPI                           |
| API Server           | Uvicorn                           |
| API Testing          | Swagger, Pytest                   |
| Model Serialization  | Pickle                            |
| Containerization     | Docker                            |
| CI/CD                | GitHub Actions                    |
| Deployment           | Render, Streamlit Community Cloud |
| Version Control      | Git, GitHub                       |

## 📂 Dataset Information

Dataset: `Mall_Customers.csv`

The dataset contains the following columns:

- CustomerID
- Genre
- Age
- Annual Income (k$)
- Spending Score (1-100)

### Features Used for Clustering

1. Annual Income (k$)
2. Spending Score (1-100)

## 🤖 Machine Learning Algorithm

**K-Means Clustering**

K-Means is an unsupervised machine learning algorithm that groups similar data points into clusters.

### Model Configuration

- Number of clusters: 5
- Initialization: K-Means++
- Random State: 42
- Number of Initializations: 10

The Elbow Method is used to help determine the appropriate number of clusters.

## 🏗️ Project Architecture

```text
                 User
                   |
                   v
           Streamlit Frontend
                   |
                   v
             FastAPI Backend
                   |
                   v
              K-Means Model
                   |
                   v
           Customer Prediction
                   |
                   v
          Cluster Visualization
```

## 📁 Project Structure

```text
Mall_Customer_Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── tests/
│   ├── conftest.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── python-tests.yml
│
├── app.py
├── api.py
├── train_model.py
├── kmeans_model.pkl
├── clustered_customers.csv
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yashwanth86868/Mall_Customer_Segmentation-Clustering.git
```

### 2. Navigate to the Project Folder

```bash
cd Mall_Customer_Segmentation-Clustering
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the FastAPI Backend

```bash
uvicorn api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Run the Streamlit Frontend

Open another terminal and run:

```bash
streamlit run app.py
```

## 🔗 API Endpoints

| Method | Endpoint           | Description                             |
| ------ | ------------------ | --------------------------------------- |
| GET    | `/`              | Check API status                        |
| POST   | `/predict`       | Predict a customer's cluster            |
| POST   | `/predict-batch` | Predict clusters for multiple customers |
| GET    | `/centroids`     | Retrieve cluster centroids              |

## 🐳 Docker Support

Build the Docker image:

```bash
docker build -t mall-customer-segmentation .
```

Run the Docker container:

```bash
docker compose up --build
```

## 🧪 Testing

Run the automated tests:

```bash
python -m pytest -v
```

GitHub Actions is configured to run automated tests when changes are pushed to the repository.

## ☁️ Deployment

**GitHub Repository:**

https://github.com/yashwanth86868/Mall_Customer_Segmentation-Clustering

**FastAPI Backend – Render:**

https://mall-customer-segmentation-clustering.onrender.com

**API Documentation:**

https://mall-customer-segmentation-clustering.onrender.com/docs

**Streamlit Frontend:**

[Add your deployed Streamlit application URL here]

## 📸 Application Screenshots

Add screenshots of your application here.

Suggested screenshots:

1. Streamlit application homepage.
2. Customer segmentation visualization.
3. Cluster analysis.
4. FastAPI Swagger documentation.
5. Prediction results.

Example:

```markdown
![Application Dashboard](screenshots/dashboard.png)

![Customer Clusters](screenshots/clusters.png)

![API Documentation](screenshots/api-docs.png)
```

## 🚀 Future Improvements

- Add customer demographic analysis.
- Experiment with different clustering algorithms.
- Improve cluster evaluation using Silhouette Score.
- Add an interactive customer recommendation system.
- Implement model monitoring and retraining.

## 👨‍💻 Author

**Yashwanth Balija**

Aspiring AI/ML Engineer

GitHub: https://github.com/yashwanth86868

---

⭐ If you find this project interesting, consider giving the repository a star!
