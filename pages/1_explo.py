import streamlit as st
import pandas as pd
import io 


st.set_page_config(page_title="✍️ Inputs", page_icon="✍️", layout="wide")

def load_ks_data():
    df = pd.read_csv('data/ks_dataset.csv', sep=",", encoding="latin1")
    return df


def clean_raw_ks_df(df):
    return df


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

    st.markdown("## Raw dataframe .info() method")
    buffer = io.StringIO()
    raw_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.markdown(
        """Notes à la volée : 
- Il y a bien des valeurs non-null dans les colonnes 13/14/15/16
        """)

    non_null_df = raw_df[(raw_df['Unnamed: 13'].notnull()) | 
                         (raw_df['Unnamed: 14'].notnull()) | 
                         (raw_df['Unnamed: 15'].notnull()) | 
                         (raw_df['Unnamed: 16'].notnull())]
    st.dataframe(non_null_df)
    st.markdown(f"""Notes à la volée:
- Le bug provient du fait que certains titres de campagne contiennent des virgules.
- Compte tenu du faible nombre de lignes concerné: {len(non_null_df)}, soit {round(len(non_null_df)/len(raw_df)*100, 2)}% du dataset, 
- Je ne vais pas prendre le temps de les cleanerpour gagner du temps mais évidemment un travail consciencieux est possible (et pas dur en +)
                """)

    clean_df = clean_raw_ks_df()

    st.markdown(raw_df.columns)
    st.markdown("## Correlation Matrix with 'state' Column")
    
    # Select numeric columns and 'state' column
    kept_cols = ['category', 'main_category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'backers', 'country', 'usd pledged']
    cols_for_corr = list(kept_cols) + ['state']
    
    # Create a copy of the dataframe with selected columns
    corr_df = raw_df[cols_for_corr].copy()
    
    # Convert 'state' to numeric (0 for failed, 1 for successful)
    corr_df['state'] = (corr_df['state'] == 'successful').astype(int)
    
    # Calculate the correlation matrix
    corr_matrix = corr_df.corr()
    
    # Display the correlation matrix
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
    
    st.markdown("""
    Notes à la volée:
    - La matrice de corrélation montre les relations linéaires entre les variables numériques et l'état du projet.
    - Une corrélation positive avec 'state' indique une association avec le succès du projet.
    - Une corrélation négative avec 'state' indique une association avec l'échec du projet.
    - Les corrélations varient de -1 (forte corrélation négative) à 1 (forte corrélation positive).
    """)
