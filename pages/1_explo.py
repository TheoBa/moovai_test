import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="✍️ Inputs", page_icon="✍️", layout="wide")

def load_ks_data():
    df = pd.read_csv('data/ks_dataset.csv', sep=",", encoding="latin1")
    return df


def clean_raw_ks_df(raw_df):
    clean_df = (raw_df.copy()
                .rename(columns=lambda x: x.strip())
                .loc[lambda x: x['currency'].isin(['GBP', 'USD', 'CAD', 'NOK', 'AUD', 'EUR', 'MXN', 'SEK', 'NZD', 'CHF,' 'DKK', 'HKD', 'SGD'])]
                .drop(columns=['ID', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'])
                )
    return clean_df


if __name__=="__main__":
    st.markdown("## Raw dataframe")
    raw_df = load_ks_data()
    st.dataframe(raw_df)
    st.markdown(
        """Notes à la volée : \n
- colonne 'ID' probablement inutile 
- deadline / launched à transfo en datetime 
- peut etre un travail d'homogénisation des valeurs $ (car currency ont différents taux) 
- Il y a plusieurs colonnes non nommées (à investiguer avant de les discard)
        """)

    st.markdown("## Raw describe() method")
    st.dataframe(raw_df.describe(include='all'))

    st.markdown("## Raw dataframe .info() method")
    buffer = io.StringIO()
    raw_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.markdown(
        """Notes à la volée : 
- Il y a bien des valeurs non-null dans les colonnes 13/14/15/16
        """)

    st.markdown("### What are those values, where do they come from?")
    non_null_df = raw_df[(raw_df['Unnamed: 13'].notnull()) | 
                         (raw_df['Unnamed: 14'].notnull()) | 
                         (raw_df['Unnamed: 15'].notnull()) | 
                         (raw_df['Unnamed: 16'].notnull())]
    st.dataframe(non_null_df)
    st.markdown("### Uniques in 'currency' column")
    st.markdown(raw_df["currency "].unique())
    st.markdown("### column names contains a space at the end")
    st.markdown(raw_df.columns)

    st.markdown(f"""Notes à la volée:
- Le bug provient du fait que certains titres de campagne contiennent des virgules.
- Compte tenu du faible nombre de lignes concerné: {len(non_null_df)}, soit {round(len(non_null_df.index)/len(raw_df.index)*100, 2)}% du dataset, 
- Je ne vais pas prendre le temps de les cleaner pour gagner du temps mais évidemment un travail consciencieux est possible (et pas dur en +)
- Après une itération je m'aperçois que certaine colonne sont au mauvais format mais ne contienne pas de valeur dans Unnamed: 13 donc je vais discriminer grace à la colonne 'currency'
- En manipulant les colonnes je m'aperçois d'un pb (espace à la fin) à cleaner aussi pour plus de clareté               
                """)

    clean_df = clean_raw_ks_df(raw_df)

    st.markdown("## Converting date columns and calculating campaign duration")

    # Convert 'launched' and 'deadline' to datetime
    clean_df['launched'] = pd.to_datetime(clean_df['launched'])
    clean_df['deadline'] = pd.to_datetime(clean_df['deadline'])

    # Calculate the time difference in days
    clean_df['timedelta'] = (clean_df['deadline'] - clean_df['launched']).dt.days

    st.write("Les colonnes 'launched' and 'deadline' sont converties en datetime pour le calcul de la colonne 'timedelta' (jours).")
    st.dataframe(clean_df.head())

    st.markdown("""
    Notes:
    - 'launched' and 'deadline' are now in datetime format
    - 'timedelta' shows the campaign duration in days
    - Negative values in 'timedelta' would indicate data errors (deadline before launch date)
    """)
    
    st.markdown("Remarque: les données de bakers & usd pledged sont des données obtenues à posteriori du lancement d'une campagne et doivent donc être écartés de la modélisation")
    
    st.markdown("## Etude de la colonne state (target de classification)")
    st.markdown("#### Uniques")
    st.markdown(clean_df.state.unique())
    st.markdown("#### Live")
    st.dataframe(clean_df[clean_df.state=='live'])
    st.markdown(f"{len(clean_df[clean_df.state=='live'].index)} campagnes sont considérées live, soit {round(len(clean_df[clean_df.state=='live'].index)/len(clean_df.index)*100, 2)}% du dataset restant")
    st.markdown("#### Undefined")
    st.dataframe(clean_df[clean_df.state=='undefined'])
    st.markdown(f"{len(clean_df[clean_df.state=='undefined'].index)} campagnes sont considérées undefined, soit {round(len(clean_df[clean_df.state=='undefined'].index)/len(clean_df.index)*100, 2)}% du dataset restant")
    st.markdown("Semble venir d'un problème que l'on peut régler puisque l'on a accès au colonnes 'goal' et 'pledged'")
    
    def undefined_state(df):
        correct_df = df.copy()
        mask = correct_df['state'] == 'undefined'
        correct_df.loc[mask, 'state'] = np.where(correct_df.loc[mask, 'pledged'] >= correct_df.loc[mask, 'goal'], 'successful', 'failed')
        return correct_df

    clean_df = undefined_state(clean_df)
    
    clean_df['label'] = np.where(clean_df['state'] == 'successful', 1, 0)
    clean_df['goal'] = clean_df['goal'].astype(float).astype(int)
    

    st.markdown(clean_df.columns)
    
    st.markdown("## Correlation Analysis et Feature Importance")

    # Select numeric columns
    numeric_columns = clean_df.select_dtypes(include=['int64', 'float64']).columns
    
    # Encode categorical columns using one-hot encoding
    categorical_columns = ['category', 'main_category', 'currency', 'country']
    encoder = OrdinalEncoder()
    encoded_features = encoder.fit_transform(clean_df[categorical_columns])
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=clean_df.index)
    
    # Combine encoded features with the original dataframe
    clean_df = pd.concat([clean_df[numeric_columns], encoded_df], axis=1)

    # Combine numeric and encoded categorical columns
    all_features = list(numeric_columns) + categorical_columns
    all_features = [col for col in all_features if col != 'label']
    numeric_columns = all_features

    # Calculate correlation matrix
    correlation_matrix = clean_df[numeric_columns + ['label']].corr()

    # Plot correlation heatmap
    st.subheader("Correlation Heatmap")
    fig_corr = plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    st.pyplot(fig_corr)


    st.markdown("## Déséquilibre du jeu de données")
    successful_campaigns = (clean_df['label'] == 1).sum()
    failed_campaigns = (clean_df['label'] == 0).sum()
    st.markdown(f"Les campagnes réussies représentent: {successful_campaigns} lignes, soit {round(successful_campaigns/len(clean_df.index)*100, 2)}% du dataset")
    st.markdown(f"Les campagnes ratées représentent: {failed_campaigns} lignes, soit {round(failed_campaigns/len(clean_df.index)*100, 2)}% du dataset")
    st.markdown("""
                Notes à la volée: pas de pb de déséquilibre""")
    
    # Feature importance using Random Forest
    st.subheader("Feature Importance")

    # Prepare the data
    X = clean_df[numeric_columns]
    y = clean_df['label']

    # Train a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X, y)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': numeric_columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot feature importance
    fig_importance = plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title("Feature Importance")
    st.pyplot(fig_importance)

    st.markdown("""
    ### Notes à la volée:
    1. Difficile de tirer des conclusions de la matrice de corrélation, tant les corrélations sont faibles avec le label.
    2. En revanche l'analyse d'importance des features ressemble plus à ce dont je m'attendais. A savoir que la quantité demandée semble beaucoup influer sur le succès ou non de la campagne
    """)
