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
    st.session_state["X_feature_importance"] = X
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
    from sklearn.preprocessing import OrdinalEncoder

    clean_df = st.session_state["X_feature_importance"]
    numeric_columns = clean_df.select_dtypes(include=['int32', 'int64', 'float64']).columns    
    categorical_columns = ['category', 'main_category', 'currency', 'country']
    encoder = OrdinalEncoder()
    encoded_features = encoder.fit_transform(clean_df[categorical_columns])
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=clean_df.index)
    encoded_df = pd.concat([clean_df[numeric_columns], encoded_df], axis=1)

    feature_importance_model = RandomForestClassifier(random_state=42)
    feature_importance_model.fit(encoded_df, st.session_state["y"])
    st.subheader("Impact on your campaign success")
    feature_importance = pd.DataFrame({
        'feature': encoded_df.columns,
        'importance': feature_importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    st.session_state["feature_importance"] = feature_importance


def plot_feature_importance():
        fig_importance = plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=st.session_state["feature_importance"])
        plt.title("Feature Importance")
        st.pyplot(fig_importance)
