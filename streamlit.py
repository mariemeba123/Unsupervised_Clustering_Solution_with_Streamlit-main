from src.visualization.visualize import cluster
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Customer Segmentation with K-Means", layout="centered")

st.markdown("""
    <style>
        /* Custom Background */
        body {
            background: linear-gradient(135deg, #f0f4f8, #c2c7d0);
            font-family: 'Arial', sans-serif;
        }
        
        .title {
            color: #1f77b4;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 20px;
        }

        .subheader {
            text-align: center;
            color: #333;
            font-size: 1.5em;
        }

        .input-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        .input-label {
            font-weight: bold;
            color: #1f77b4;
        }

        .stButton button {
            background-color: #1f77b4;
            color: white;
            padding: 12px 24px;
            font-size: 1.1em;
            border-radius: 5px;
            border: none;
        }

        .stButton button:hover {
            background-color: #155a8a;
        }

        .result-text {
            font-size: 1.2em;
            color: #28a745;
            text-align: center;
            margin-top: 20px;
        }

        .image-container {
            text-align: center;
            margin-top: 30px;
        }

        .image-container img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .stTextInput, .stNumberInput, .stSelectbox, .stTextArea {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    .stTextInput:hover, .stNumberInput:hover, .stSelectbox:hover, .stTextArea:hover {
        border-color: #1f77b4;
    }
    .stSelectbox select, .stNumberInput input {
        background-color: #ffffff;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    """
    <div style="background-color:#1f77b4;padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Customer Segmentation with K-Means</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="font-size:16px;color:#333;text-align:center;margin-top:10px;">
    This application predicts your customer segment.
    </p>
    """,
    unsafe_allow_html=True
)
# Load the K-Means model
with open("models/Kmodel.pkl", "rb") as model_file:
    kmeans = pickle.load(model_file)

# Load data for visualization
df = pd.read_csv("data/raw/mall_customers.csv")

# User interface for entering features

with st.form("prediction_form", clear_on_submit=True):
    st.subheader("Enter Your Informations:")
    age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30, label_visibility="visible")
    income = st.number_input("Annual Income (k$)",  step=1, value=50, label_visibility="visible")
    spending = st.number_input("Spending Score", step=1, value=50, label_visibility="visible")
    
    submitted = st.form_submit_button("Predict My Cluster")
   

# Prediction using the model
if submitted:
    # Create user data frame
    user_data = pd.DataFrame([[age, income, spending]], columns=["Age", "Annual_Income", "Spending_Score"])
    
    # Predict the cluster
    cluster_pred = kmeans.predict(user_data)[0]
    st.markdown(f'<div class="result-text">You belong to <strong>Cluster {cluster_pred}</strong></div>', unsafe_allow_html=True)
    
    # Display silhouette plot
    st.image("silhouette.png", caption="Silhouette Score for Cluster Quality", use_container_width=True) 

    # Add the user point to the original dataframe for visualization
    df["Cluster"] = kmeans.labels_
    cluster(df,income,spending)
    # Additional cluster image visualization
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("cluster.png", caption="Cluster Visualization", use_container_width=True) 
    st.markdown('</div>', unsafe_allow_html=True)

# --- Feature explanation ---
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:10px 20px;border-radius:10px;margin-top:30px">
    <h4 style="color:#1f77b4;"> How does this work?</h4>
    <p style="color:#333;">
   We used a machine learning (K-Means) model to find out your customer segment, the features used in this prediction are ranked by relative
    importance below.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

