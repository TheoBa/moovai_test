import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, Normalizer, StandardScaler, RobustScaler


SCALERS = {
     "normalization": Normalizer(),
     "standardization": StandardScaler(),
     "robust": RobustScaler(),
}


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


def encode_categorical_columns(df, categorical_columns):
    if categorical_columns == "None":
         return df
    encoder = OrdinalEncoder()
    encoded_features = encoder.fit_transform(df[categorical_columns])
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
    numeric_columns = list(set(df.columns).difference(set(categorical_columns)))
    encoded_df = pd.concat([df[numeric_columns], encoded_df], axis=1)

    return encoded_df


def scaling(df, columns_to_scale, scaler):
    if scaler == "None":
         return df
    not_scaled_columns = list(set(df.columns).difference(set(columns_to_scale)))
    scaler = SCALERS[scaler]
    scaled_data = scaler.fit_transform(df[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
    scaled_df = scaled_df.merge(df[not_scaled_columns], on=df.index, how='left').drop(columns=['key_0'])
    return scaled_df


def feature_eng(clean_df):
    feature_df = (
         clean_df
         .assign(
              # Convert 'launched' and 'deadline' to datetime 
              launched=lambda df: pd.to_datetime(df['launched']),
              deadline=lambda df: pd.to_datetime(df['deadline']),
              # Compute timedelta
              timedelta=lambda df: (df['deadline'] - df['launched']).dt.days,
              # Extract year and month of campaign launch
              year=lambda x: x["launched"].dt.year,
              month=lambda x: x["launched"].dt.month,
              # Define label
              label=lambda df: np.where(df['state'] == 'successful', 1, 0)
         )
         #.pipe(encode_categorical_columns, categorical_columns)      
         #.pipe(scaling, columns_to_scale, scaler)
         
         )
    return feature_df
