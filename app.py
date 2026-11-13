# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# # Page Configuration
# st.set_page_config(
#     page_title="Mall Customer Segmentation",
#     page_icon="🛍️",
#     layout="wide"
# )

# # Load Model
# @st.cache_resource
# def load_model():
#     with open("kmeans_model.pkl", "rb") as file:
#         return pickle.load(file)

# model = load_model()

# # Load Dataset
# @st.cache_data
# def load_data():
#     return pd.read_csv("data/Mall_Customers.csv")

# dataset = load_data()

# # Predict Existing Customer Clusters
# X = dataset.iloc[:, [3, 4]].values
# dataset["cluster"] = model.predict(X)

# # Dashboard Title
# st.title("🛍️ Mall Customer Segmentation")

# st.markdown(
#     "### AI-powered Customer Segmentation using K-Means Clustering"
# )

# st.divider()

# # Sidebar
# st.sidebar.header("Customer Details")

# income = st.sidebar.number_input(
#     "Annual Income (k$)",
#     min_value=0.0,
#     max_value=200.0,
#     value=50.0
# )

# spending = st.sidebar.slider(
#     "Spending Score",
#     min_value=1,
#     max_value=100,
#     value=50
# )

# predict_button = st.sidebar.button(
#     "Predict Customer Cluster",
#     use_container_width=True
# )

# # Dashboard Metrics
# col1, col2, col3 = st.columns(3)

# col1.metric(
#     "Total Customers",
#     len(dataset)
# )

# col2.metric(
#     "Total Clusters",
#     model.n_clusters
# )

# col3.metric(
#     "Average Income",
#     f"${dataset.iloc[:, 3].mean():.2f}k"
# )

# st.divider()

# # Customer Prediction
# st.subheader("🎯 Customer Cluster Prediction")

# if predict_button:

#     new_customer = np.array([[income, spending]])

#     prediction = model.predict(new_customer)[0]

#     st.success(
#         f"This customer belongs to Cluster {prediction + 1}"
#     )

#     st.info(
#         f"Annual Income: ${income}k | Spending Score: {spending}"
#     )

# # Cluster Visualization
# st.divider()

# st.subheader("📊 Customer Segmentation Visualization")

# fig, ax = plt.subplots(figsize=(10, 6))

# for i in range(model.n_clusters):

#     cluster_data = dataset[dataset["cluster"] == i]

#     ax.scatter(
#         cluster_data.iloc[:, 3],
#         cluster_data.iloc[:, 4],
#         s=70,
#         label=f"Cluster {i+1}"
#     )

# # Centroids
# ax.scatter(
#     model.cluster_centers_[:, 0],
#     model.cluster_centers_[:, 1],
#     s=250,
#     c="black",
#     marker="X",
#     label="Centroids"
# )

# ax.set_xlabel("Annual Income (k$)")
# ax.set_ylabel("Spending Score (1-100)")
# ax.set_title("Mall Customer Segmentation")

# ax.legend()
# ax.grid(True)

# st.pyplot(fig)

# plt.close(fig)

# # Cluster Analysis
# st.divider()

# st.subheader("📈 Cluster Analysis")

# cluster_analysis = dataset.groupby("cluster").agg({
#     dataset.columns[3]: "mean",
#     dataset.columns[4]: "mean",
#     "CustomerID": "count"
# }).rename(columns={
#     dataset.columns[3]: "Average Annual Income",
#     dataset.columns[4]: "Average Spending Score",
#     "CustomerID": "Customer Count"
# })

# st.dataframe(
#     cluster_analysis,
#     use_container_width=True
# )

# # Dataset Preview
# st.divider()

# st.subheader("📋 Customer Dataset")

# st.dataframe(
#     dataset,
#     use_container_width=True
# )

# st.success("Customer Segmentation Dashboard is Running!")







import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🛍️",
    layout="wide"
)

API_URL = https://mall-customer-segmentation-clustering.onrender.com

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #1f2937;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🛍️ Customer Segmentation AI")

st.markdown(
    "### Intelligent Customer Analytics using K-Means Clustering"
)

st.caption(
    "Python | Machine Learning | FastAPI | Streamlit | Plotly"
)

st.divider()

# ==========================================
# 4. API CONNECTION
# ==========================================
    
def check_api():
    

    try:
        response = requests.get(
            f"{API_URL}/",
            timeout=60
        )

        response.raise_for_status()

        return True

    except requests.exceptions.RequestException as error:

        st.error(f"API Connection Error: {error}")

    return False

   

if not check_api():

    st.error(
        "FastAPI backend is not running. "
        "Please start it using: uvicorn api:app --reload"
    )

    st.stop()

st.success("🟢 FastAPI Backend Connected Successfully!")

# ==========================================
# 5. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv("data/Mall_Customers.csv")

try:

    dataset = load_data()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()

income_col = "Annual Income (k$)"
spending_col = "Spending Score (1-100)"

required_columns = [
    "CustomerID",
    "Genre",
    "Age",
    income_col,
    spending_col
]

missing_columns = [
    col for col in required_columns
    if col not in dataset.columns
]

if missing_columns:

    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

# ==========================================
# 6. PREPARE CUSTOMER DATA
# ==========================================

X = dataset[
    [income_col, spending_col]
].values

# ==========================================
# 7. GET BATCH PREDICTIONS FROM FASTAPI
# ==========================================

@st.cache_data
def get_batch_predictions(customer_data):

    response = requests.post(
        f"{API_URL}/predict-batch",
        json=customer_data,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if "predictions" not in result:
        raise ValueError(
            f"Unexpected batch API response: {result}"
        )

    return result["predictions"]

customer_data = [
    {
        "annual_income": float(income),
        "spending_score": float(spending)
    }
    for income, spending in X
]

try:

    predictions = get_batch_predictions(
        customer_data
    )

    if len(predictions) != len(dataset):
        st.error("Batch prediction count does not match dataset.")
        st.stop()

    # Convert API cluster numbers (1-5)
    # into internal cluster indexes (0-4)

    dataset["cluster"] = [
        int(cluster) - 1
        for cluster in predictions
    ]

except Exception as error:

    st.error(f"Batch prediction failed: {error}")
    st.stop()

# ==========================================
# 8. GET CLUSTER CENTROIDS FROM FASTAPI
# ==========================================

@st.cache_data
def get_centroids():

    response = requests.get(
        f"{API_URL}/centroids",
        timeout=10
    )

    response.raise_for_status()

    result = response.json()

    if "centroids" not in result:
        raise ValueError(
            f"Unexpected centroid API response: {result}"
        )

    return result["centroids"]

try:

    centers = np.array(
        get_centroids(),
        dtype=float
    )

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Invalid centroid dimensions.")

    if not np.isfinite(centers).all():
        raise ValueError("Centroids contain invalid values.")

    if len(centers) == 0:
        raise ValueError("No centroids returned by API.")

    if not dataset["cluster"].between(
        0, len(centers) - 1
    ).all():
        raise ValueError("API returned an invalid cluster number.")

except Exception as error:

    st.error(f"Centroid loading failed: {error}")
    st.stop()

# ==========================================
# 9. CUSTOMER SEGMENT NAMES
# ==========================================

income_median = np.median(centers[:, 0])
spending_median = np.median(centers[:, 1])

segment_names = {}

for i, center in enumerate(centers):

    income = center[0]
    spending = center[1]

    if income > income_median and spending > spending_median:

        name = "Premium Customers"

    elif income > income_median and spending <= spending_median:

        name = "Careful High-Income"

    elif income <= income_median and spending > spending_median:

        name = "Enthusiastic Shoppers"

    elif income < income_median and spending < spending_median:

        name = "Budget Conscious"

    else:

        name = "Average Customers"

    segment_names[i] = f"Cluster {i+1} - {name}"

dataset["Segment"] = dataset["cluster"].map(
    segment_names
)

# ==========================================
# 10. DASHBOARD METRICS
# ==========================================

st.header("📊 Business Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(dataset)
)

col2.metric(
    "Total Clusters",
    len(centers)
)

col3.metric(
    "Average Income",
    f"${dataset[income_col].mean():.1f}k"
)

col4.metric(
    "Average Spending",
    f"{dataset[spending_col].mean():.1f}/100"
)

st.divider()

# ==========================================
# 11. CUSTOMER PREDICTION
# ==========================================

st.header("🎯 Predict Customer Segment")

left, right = st.columns(2)

with left:

    income = st.number_input(
        "Annual Income (k$)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=1.0
    )

with right:

    spending = st.slider(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

if st.button(
    "Predict Customer Segment 🚀",
    use_container_width=True
):

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json={
                "annual_income": income,
                "spending_score": spending
            },
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        # Validate API response

        if "predicted_cluster" not in result:

            st.error(
                f"Unexpected API response: {result}"
            )

        else:

            cluster = int(
                result["predicted_cluster"]
            ) - 1

            if cluster < 0 or cluster >= len(centers):

                st.error(
                    "Invalid cluster number received from API."
                )

            else:

                st.success(
                    f"Predicted Segment: {segment_names[cluster]}"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Cluster Number",
                    cluster + 1
                )

                col2.metric(
                    "Annual Income",
                    f"${income}k"
                )

                col3.metric(
                    "Spending Score",
                    spending
                )

                centroid = centers[cluster]

                st.info(
                    f"Nearest Cluster Centroid: "
                    f"Income ${centroid[0]:.1f}k | "
                    f"Spending Score {centroid[1]:.1f}"
                )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Prediction API Error: {error}"
        )

    except (ValueError, TypeError, KeyError) as error:

        st.error(
            f"Prediction response error: {error}"
        )

st.divider()

# ==========================================
# 12. INTERACTIVE CUSTOMER VISUALIZATION
# ==========================================

st.header("📈 Customer Segmentation Visualization")

fig = px.scatter(
    dataset,
    x=income_col,
    y=spending_col,
    color="Segment",
    hover_data=[
        "CustomerID",
        "Genre",
        "Age"
    ],
    title="Customer Groups by Income and Spending",
    template="plotly_white"
)

# Display cluster centroids

fig.add_trace(
    go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode="markers",
        name="Centroids",
        marker=dict(
            symbol="x",
            size=18,
            color="black",
            line=dict(width=3)
        )
    )
)

fig.update_layout(
    height=550,
    xaxis_title="Annual Income (k$)",
    yaxis_title="Spending Score",
    legend_title="Customer Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ==========================================
# 13. CLUSTER ANALYSIS
# ==========================================

st.header("📊 Customer Cluster Analysis")

analysis = dataset.groupby("Segment").agg(

    Customer_Count=("CustomerID", "count"),

    Average_Income=(
        income_col,
        "mean"
    ),

    Average_Spending=(
        spending_col,
        "mean"
    )

).reset_index()

st.dataframe(
    analysis.round(2),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 14. CUSTOMER DATASET
# ==========================================

st.divider()

st.header("📋 Customer Dataset")

st.dataframe(
    dataset,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 15. DOWNLOAD CUSTOMER REPORT
# ==========================================

st.header("📥 Download Customer Report")

csv = dataset.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Segmentation Report 📥",
    data=csv,
    file_name="customer_segmentation_report.csv",
    mime="text/csv",
    use_container_width=True
)

# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <center>
        <h4>Customer Segmentation AI</h4>
        <p>
        Developed using Python, Scikit-learn,
        FastAPI, Streamlit and Plotly.
        </p>
        <p>Machine Learning Project</p>
    </center>
    """,
    unsafe_allow_html=True
)