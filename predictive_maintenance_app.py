import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

def load_model_and_scaler(model_path="xgboost_model.pkl", scaler_path="scaler.pkl"):
    """Load the pre-trained XGBoost model and scaler from pickle files."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model or scaler file not found. Please ensure {model_path} and {scaler_path} exist.")
        return None, None

def load_data(file_path="enhanced_predictive_maintenance_dataset.csv"):
    """Load dataset for evaluation."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.warning(f"{file_path} not found. Please upload a dataset.")
        return None

def preprocess_data(df, features):
    """Preprocess dataset for evaluation."""
    X = df[features]
    y = df['failure']
    X = X.fillna(X.mean())
    return X, y

def evaluate_model(model, scaler, X, y):
    """Evaluate the model and return metrics."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    report = classification_report(y, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y, y_pred_proba)
    return report, roc_auc

def plot_feature_importance(model, features):
    """Plot feature importance."""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title('XGBoost Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    return fig

def main():
    st.title("Predictive Maintenance Dashboard")
    st.markdown("This app predicts equipment failures using an XGBoost model and provides maintenance recommendations.")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    # Define features
    features = ['temperature', 'vibration', 'pressure', 'runtime_hours', 
                'current', 'noise_level', 'oil_quality', 'wear_rate']
    
    # Sidebar for dataset upload
    st.sidebar.header("Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    else:
        df = load_data()
    
    # Model Evaluation Section
    if df is not None:
        st.header("Model Evaluation")
        X, y = preprocess_data(df, features)
        report, roc_auc = evaluate_model(model, scaler, X, y)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report")
            st.json(report)
        with col2:
            st.subheader("ROC-AUC Score")
            st.metric(label="ROC-AUC", value=f"{roc_auc:.4f}")
        
        st.subheader("Feature Importance")
        fig = plot_feature_importance(model, features)
        st.pyplot(fig)
    
    # Prediction Section
    st.header("Predict Failure Probability")
    st.markdown("Enter sensor data to predict equipment failure probability.")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature (°F)", min_value=0.0, max_value=200.0, value=78.0)
        vibration = st.number_input("Vibration (mm/s)", min_value=0.0, max_value=10.0, value=0.65)
        pressure = st.number_input("Pressure (psi)", min_value=0.0, max_value=500.0, value=115.0)
        runtime_hours = st.number_input("Runtime Hours", min_value=0.0, max_value=10000.0, value=1300.0)
    with col2:
        current = st.number_input("Current (Amps)", min_value=0.0, max_value=50.0, value=17.0)
        noise_level = st.number_input("Noise Level (dB)", min_value=0.0, max_value=120.0, value=70.0)
        oil_quality = st.number_input("Oil Quality (%)", min_value=0.0, max_value=100.0, value=65.0)
        wear_rate = st.number_input("Wear Rate (mm/month)", min_value=0.0, max_value=1.0, value=0.03)
    
    if st.button("Predict"):
        new_data = [temperature, vibration, pressure, runtime_hours, 
                    current, noise_level, oil_quality, wear_rate]
        new_data_df = pd.DataFrame([new_data], columns=features)
        new_data_scaled = scaler.transform(new_data_df)
        failure_prob = model.predict_proba(new_data_scaled)[0][1]
        
        st.subheader("Prediction Result")
        st.metric(label="Failure Probability", value=f"{failure_prob:.2%}")
        
        if failure_prob > 0.7:
            st.error("Recommendation: Schedule maintenance immediately.")
        elif failure_prob > 0.3:
            st.warning("Recommendation: Monitor closely and consider maintenance soon.")
        else:
            st.success("Recommendation: Equipment appears stable.")

if __name__ == "__main__":
    main()