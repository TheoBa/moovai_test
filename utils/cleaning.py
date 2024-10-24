import streamlit as st
import pandas as pd
import numpy as np


def load_ks_data():
    df = pd.read_csv('data/ks_dataset.csv', sep=",", encoding="latin1")
    return df


def undefined_state(df):
        correct_df = df.copy()
        mask = correct_df['state'] == 'undefined'
        correct_df.loc[mask, 'state'] = np.where(correct_df.loc[mask, 'pledged'] >= correct_df.loc[mask, 'goal'], 'successful', 'failed')
        return correct_df


def clean_raw_ks_df(raw_df):
    clean_df = (
        raw_df.copy()
        .rename(columns=lambda x: x.strip())
        .loc[lambda x: x['currency'].isin(['GBP', 'USD', 'CAD', 'NOK', 'AUD', 'EUR', 'MXN', 'SEK', 'NZD', 'CHF,' 'DKK', 'HKD', 'SGD'])]
        .drop(columns=['ID', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'])
        .pipe(undefined_state)
        .assign(goal=lambda df: df['goal'].astype(float).astype(int))
    )
    return clean_df


def feature_eng(clean_df):
    feature_df = (clean_df
                  .assign(
                      # Convert 'launched' and 'deadline' to datetime 
                      launched=lambda df: pd.to_datetime(df['launched']),
                      deadline=lambda df: pd.to_datetime(df['deadline']),
                      # Compute timedelta
                      timedelta=lambda df: (df['deadline'] - df['launched']).dt.days,
                      # Define label
                      label=lambda df: np.where(df['state'] == 'successful', 1, 0)
                      )                  
    )
    return feature_df