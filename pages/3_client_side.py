import streamlit as st
import pandas as pd
from utils.inputs import CAMPAIGN_INPUTS
from utils.client_model import init_client_model, train_client_model, get_feature_importance, plot_feature_importance
from utils.cleaning import feature_eng_client_input

st.set_page_config(page_title="Client side", page_icon="✍️", layout="wide")

st.title("Kickstarter campaign optimizer")
st.markdown("This tool will help you better understand how to launch successful campaigns.")
st.markdown("It will also provide you with a success rate of a given campaign.")


def initialize_inputs():
    for key, value in CAMPAIGN_INPUTS.items():
        session_key = f'_{key}'
        if session_key not in st.session_state:
            st.session_state[session_key] = value


def get_campaign_information():
    st.subheader("Provide your campaign informations")

    user_inputs = {}
    with st.form("Campaign inputs form"):
        buffer1, col, buffer2 = st.columns([1, 3, 1])
        with col:
            for key, input_properties in CAMPAIGN_INPUTS.items():
                input_type = input_properties[0]
                options = input_properties[1] 
                if input_type == 'text':
                    user_inputs[key] = st.selectbox(label=key, options=options)
                elif input_type == 'date':
                    user_inputs[key] = st.date_input(label=key)
                elif input_type == 'int':
                    user_inputs[key] = st.number_input(label=key, value=options)
        
        footer_cols = st.columns([5,1])
        with footer_cols[1]:
            submitted = st.form_submit_button()
        
        # Update the session state with the new DataFrame
        if submitted:
            for key, value in CAMPAIGN_INPUTS.items():
                st.session_state[f"_{key}"] = user_inputs[key]
            st.session_state["client_inputs"] = pd.DataFrame([user_inputs])[list(CAMPAIGN_INPUTS.keys())]
            
            st.session_state["clean_client_inputs"] = feature_eng_client_input(st.session_state["client_inputs"])

            encode_client_inputs()
            scale_inputs()
            get_prediction()


def encode_client_inputs():
    encoded_dict = {}
    for col in st.session_state["X"].columns:
        if col in list(st.session_state["clean_client_inputs"].columns):
            encoded_dict[col] = st.session_state["clean_client_inputs"].loc[0, col]
        else:
            encoded_dict[col] = 0
    
    for col in ['category', 'main_category', 'currency', 'country']:
        encoded_dict[f"{col}_{st.session_state['clean_client_inputs'].loc[0, col]}"] = 1

    st.session_state["model_ready_user_inputs"] = pd.DataFrame([encoded_dict])[st.session_state["X"].columns]


def scale_inputs():
    st.session_state["scaled_user_inputs"] = st.session_state["scaler"].transform(st.session_state["model_ready_user_inputs"])


def get_prediction():
    pred_proba = st.session_state["model"].predict_proba(st.session_state["scaled_user_inputs"])
    st.metric(label="Taux de réussite", value=f"{int(pred_proba[0][1]*10000)/100} %")


if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False

if not(st.session_state["model_trained"]):
    init_client_model()
    train_client_model()
    st.session_state["model_trained"] = True

initialize_inputs()
get_campaign_information()

FE_button = st.button("Plot feature_importance")
if FE_button:
    get_feature_importance()
    plot_feature_importance()







