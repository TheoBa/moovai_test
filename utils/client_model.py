import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.cleaning import load_ks_data, clean_raw_ks_df, feature_eng
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier


def init_client_model():
    # Define features and target
    features = ['goal', 'timedelta', 'category', 'main_category', 'currency', 'country', 'year', 'month']
    categorical_features = ['category', 'main_category', 'currency', 'country']
    target = 'label'
    feature_df = (
        load_ks_data()
        .pipe(clean_raw_ks_df)
        .pipe(feature_eng)
    )
    # Prepare the data
    X = feature_df[features]
    y = feature_df[target]
    # Perform one-hot encoding for categorical variables
    X = pd.get_dummies(X, columns=categorical_features)
    # Scaling
    scaler = RobustScaler()
    
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["scaler"] = scaler.fit(X)
    st.session_state["X_scaled"] = st.session_state["scaler"].transform(X)


def train_client_model():
    # Initialize models
    model = XGBClassifier(random_state=42)

    # Train the model
    model.fit(st.session_state["X_scaled"], st.session_state["y"])
    st.session_state["model"] = model


def get_feature_importance():
    # Feature importance for Random Forest 
    feature_importance_model = RandomForestClassifier(random_state=42)
    feature_importance_model.fit(st.session_state["X"], st.session_state["y"])
    st.subheader("Feature Importance from the Random Forest model")
    feature_importance = pd.DataFrame({
        'feature': st.session_state["X"].columns,
        'importance': feature_importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    st.session_state["feature_importance"] = feature_importance


def plot_feature_importance():
        fig_importance = plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=st.session_state["feature_importance"])
        plt.title("Feature Importance")
        st.pyplot(fig_importance)
