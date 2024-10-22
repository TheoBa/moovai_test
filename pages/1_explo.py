import streamlit as st
import pandas as pd


st.set_page_config(page_title="✍️ Inputs", page_icon="✍️", layout="wide")

def load_ks_data():
    df = pd.read_csv('data/ks_dataset.csv')
    return df


if __name__=="__main__":
    st.markdown("WIP")