import streamlit as st
import pandas as pd
import numpy as np
from utils.cleaning import load_ks_data, clean_raw_ks_df, feature_eng

st.set_page_config(page_title="Modelisation", page_icon="✍️", layout="wide")


feature_df = (
    load_ks_data()
    .pipe(clean_raw_ks_df)
    .pipe(feature_eng)
)

st.dataframe(feature_df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define features and target
features = ['goal', 'timedelta', 'category', 'main_category', 'currency', 'country']
target = 'label'

# Prepare the data
X = feature_df[features]
y = feature_df[target]

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['category', 'main_category', 'currency', 'country'])

# Split the data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1) # X_test : 10% ; X_train_val: 90%
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.15 * 10/9) # X_val is 15% ; X_train: 75%

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    st.subheader(f"Training {name}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on validation set
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_val_pred)
    
    # Display results
    st.write(f"Validation Accuracy: {accuracy:.4f}")
    st.write("Classification Report:")
    st.text(classification_report(y_val, y_val_pred))

# Feature importance for Random Forest 

st.subheader("Feature Importance from the Random Forest model")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': models["Random Forest"].feature_importances_
}).sort_values('importance', ascending=False)
st.dataframe(feature_importance)

