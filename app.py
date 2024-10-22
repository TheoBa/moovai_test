import streamlit as st


st.set_page_config(
    page_title='Homepage', 
    page_icon='💸', 
    layout="wide", 
    initial_sidebar_state="collapsed"
    )

def welcome_page():
    st.markdown(
    """
    # Test technique MoovAI
    
    **👈 Navigue au travers des questions dans la sidebar** pour accéder au code et visualisations utilisés
    \n
    #### **✍️ Q1**
    #### **📈 Q2**
    #### **📊 Q3**
    #### **📊 Q4**
    """
    )



if __name__=="__main__":
    welcome_page()