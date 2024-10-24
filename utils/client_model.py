import streamlit as st
import pandas as pd
from utils.cleaning import load_ks_data, clean_raw_ks_df, feature_eng
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier


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
X_train_scaled = scaler.fit_transform(X)


def train_client_model():
    # Initialize models
    model = XGBClassifier(random_state=42)

    # Train the model
    model.fit(X_train_scaled, y)
    return model


    # # Make predictions on validation set
    # y_val_pred = model.predict(X_val_scaled)


def get_feature_importance():
    # Feature importance for Random Forest 
    feature_importance_model = RandomForestClassifier(random_state=42)
    feature_importance_model.fit(X, y)
    st.subheader("Feature Importance from the Random Forest model")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    return feature_importance

