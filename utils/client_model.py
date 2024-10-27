import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.cleaning import load_ks_data, clean_raw_ks_df, feature_eng
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ClientModel:
    def __init__(self):
        self.features = ['goal', 'timedelta', 'category', 'main_category', 'currency', 'country', 'year', 'month']
        self.categorical_features = ['category', 'main_category', 'currency', 'country']
        self.target = 'label'
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = RobustScaler()
        self.model = XGBClassifier(random_state=42)
        self.feature_importance = None

    def load_and_prepare_data(self):
        feature_df = (
            load_ks_data()
            .pipe(clean_raw_ks_df)
            .pipe(feature_eng)
        )
        self.X = feature_df[self.features]
        self.y = feature_df[self.target]
        self.X_feature_importance = self.X.copy()
        
        # Perform one-hot encoding for categorical variables
        self.X = pd.get_dummies(self.X, columns=self.categorical_features)
        
        # Scaling
        self.X_scaled = self.scaler.fit_transform(self.X)

    def train_model(self):
        self.model.fit(self.X_scaled, self.y)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(self.scaler.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def get_feature_importance(self):
        encoder = OrdinalEncoder()
        numeric_columns = self.X_feature_importance.select_dtypes(include=['int32', 'int64', 'float64']).columns
        encoded_features = encoder.fit_transform(self.X_feature_importance[self.categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(self.categorical_features)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=self.X_feature_importance.index)
        encoded_df = pd.concat([self.X_feature_importance[numeric_columns], encoded_df], axis=1)

        feature_importance_model = RandomForestClassifier(random_state=42)
        feature_importance_model.fit(encoded_df, self.y)
        self.feature_importance = pd.DataFrame({
            'feature': encoded_df.columns,
            'importance': feature_importance_model.feature_importances_
        }).sort_values('importance', ascending=False)

    def plot_feature_importance(self):
        if self.feature_importance is None:
            self.get_feature_importance()
        
        fig_importance = plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=self.feature_importance)
        plt.title("Feature Importance")
        return fig_importance

# Helper functions to integrate with Streamlit
def init_client_model():
    st.session_state["client_model"] = ClientModel()
    st.session_state["client_model"].load_and_prepare_data()

def train_client_model():
    st.session_state["client_model"].train_model()

def get_feature_importance():
    st.session_state["client_model"].get_feature_importance()
    st.subheader("Impact on your campaign success")
    st.dataframe(st.session_state["client_model"].feature_importance)

def plot_feature_importance():
    fig = st.session_state["client_model"].plot_feature_importance()
    st.pyplot(fig)
