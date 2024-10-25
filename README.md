# moovai_test
Repo test technique pour DS à MoovAI

# How to use the repo
poetry shell
streamlit run app.py

The explo tab shows visualization I used while exploring the dataset
The modelisation tab shows the experiences i ran while exploring the modelisation of the problem
The client side tab shows an **utterly** simple front end solution I would provide to the client



# Question #1
Problèmes de qualité des données raw kickstarter
- Colonnes _Unnamed: x_ après loading du csv en dataframe: certains titres de campagnes contiennent des virgules ce qui est génant avec le csv. Compte tenu du faible nombre de lignes concernées, ces lignes sont supprimées
- Le nom des colonne contient un espace ' ' à la fin, cleanable simplement
- Colonnes inutiles : _ID_
- Format datetime dans les colonnes _launched_ et _deadline_
- Valeurs numéraires non homogènes dans la colonne _goal_ (ça je ne l'ai pas traité et maintenant que je rédige les réponses à la fin je me rends compte que j'aurais du !)
- Certaines campagnes sont 'undefined' dans la colonne _state_, possible de le reconstruire avec les autres colonnes (_goal_ et _pledged_)


# Question #2
Identification d'insights sur la contribution au succès ou non des campagnes
- feature importance
![plot](./data/feature_importance.png)
- histograms
![plot](data/hist_main_category.png)
- distribution of campaigns features
![plot](/data/distribution_of_goal_amount_and_timedelta.png)

Concernant les facteurs de confusions je me rends compte lors de la rédaction des réponses que la non-homogénéité des valeurs numéraires dans la colonne _goal_ a certainement nuit au perf du modèle et nécessite un travail.
La feature _timedelta_ a présenté plus d'importance dans l'étude de feature importance que _launched_ et _dealine_ séparement et a donc naturellement fait l'objet d'une feature.
