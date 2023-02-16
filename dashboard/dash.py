import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Chargement du dataset
df = pd.read_csv("app_test_dashboard_with_prediction")

# Configuration de la page
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

# Sidebar
st.sidebar.header("Veuillez appliquer un filtre:")

contract_type = st.sidebar.multiselect(
    "Choisir le type de contrat:",
    options=df["NAME_CONTRACT_TYPE"].unique(),
    default=df["NAME_CONTRACT_TYPE"].unique()
)

sexe = st.sidebar.multiselect(
    "Choisir le sexe:",
    options=df["CODE_GENDER"].unique(),
    default=df["CODE_GENDER"].unique()
)

civil_status = st.sidebar.multiselect(
    "Choisir l'état civil:",
    options=df["NAME_FAMILY_STATUS"].unique(),
    default=df["NAME_FAMILY_STATUS"].unique()
)

habitation_type = st.sidebar.multiselect(
    "Choisir le type d'habitation:",
    options=df["NAME_HOUSING_TYPE"].unique(),
    default=df["NAME_HOUSING_TYPE"].unique()
)

nombre_enfants = st.sidebar.multiselect(
    "Choisir le nombre d'enfants:",
    options=df["CNT_CHILDREN"].unique(),
    default=df["CNT_CHILDREN"].unique()
)

df_selection = df.query(
    "NAME_CONTRACT_TYPE == @contract_type & CODE_GENDER == @sexe & NAME_FAMILY_STATUS == @civil_status & NAME_HOUSING_TYPE == @habitation_type & CNT_CHILDREN == @nombre_enfants"
)

st.dataframe(df_selection)

# Page principale
st.title(":bar_chart: Informations clients")
st.markdown("##")

# Informations
nombre_clients = int(df_selection.shape[0])
average_age = round((abs(df_selection["DAYS_BIRTH"])/365).mean(), 1)
average_prediction = round(100 * df_selection["Prediction : 1"].mean(), 2)
family = ":family:"

colonne_gauche, colonne_droite = st.columns(2)
with colonne_gauche:
    st.subheader("Nombre de clients :")
    st.subheader(f"{nombre_clients}")
with colonne_droite:
    st.subheader("Âge moyen des clients :")
    st.subheader(f"{family} {average_age} ans")


# Prédictions selon le type de contrat
prediction_contrat = (df_selection.groupby(by="NAME_CONTRACT_TYPE")["Prediction : 1"].mean().sort_values()
)

graph_contrat = px.bar(
    prediction_contrat,
    x="Prediction : 1",
    y=prediction_contrat.index,
    orientation="h",
    title="<b>Prédictions du modèle : risque de non-remboursement selon le type de contrat",
    color_discrete_sequence=["#0083B8"] * len(prediction_contrat),
    template="plotly"
)

st.plotly_chart(graph_contrat)

# Prédictions selon le sexe
prediction_sexe = (df_selection.groupby(by="CODE_GENDER")["Prediction : 1"].mean().sort_values()
)

graph_sexe = px.bar(
    prediction_sexe,
    x="Prediction : 1",
    y=prediction_sexe.index,
    orientation="h",
    title="<b>Prédictions du modèle : risque de non-remboursement selon le sexe du client",
    color_discrete_sequence=["#0083B8"] * len(prediction_sexe),
    template="plotly"
)

st.plotly_chart(graph_sexe)

# Prédictions selon l'état civil
prediction_civil = (df_selection.groupby(by="NAME_FAMILY_STATUS")["Prediction : 1"].mean().sort_values()
)

graph_civil = px.bar(
    prediction_civil,
    x="Prediction : 1",
    y=prediction_civil.index,
    orientation="h",
    title="<b>Prédictions du modèle : risque de non-remboursement selon l'état civil du client",
    color_discrete_sequence=["#0083B8"] * len(prediction_civil),
    template="plotly"
)

st.plotly_chart(graph_civil)

# Prédictions selon le type d'habitation
prediction_habitation = (df_selection.groupby(by="NAME_HOUSING_TYPE")["Prediction : 1"].mean().sort_values()
)

graph_habitation = px.bar(
    prediction_habitation,
    x="Prediction : 1",
    y=prediction_habitation.index,
    orientation="h",
    title="<b>Prédictions du modèle : risque de non-remboursement selon le type d'habitation du client",
    color_discrete_sequence=["#0083B8"] * len(prediction_habitation),
    template="plotly"
)

st.plotly_chart(graph_habitation)

# Affichage 1 : infos client

base_url = 'https://projet-7-toth-maxime.herokuapp.com/infos_client'
user_input = st.text_input("Entrez l'ID Client pour afficher les infos :")
full_url = base_url + '?id_client=' + user_input
response = requests.get(full_url)
if response.status_code == 200:
    data = response.json()
    st.write(data)
else:
    st.write('Erreur :', response.status_code)

# Affichage 2 : Prédiction

base_url = 'https://projet-7-toth-maxime.herokuapp.com/infos_client'
user_input = st.text_input("Entrez l'ID Client pour afficher la prédiction :")
full_url = base_url + '?id_client=' + user_input
response = requests.get(full_url)
if response.status_code == 200:
    data = response.json()
    st.write(data)
else:
    st.write('Erreur :', response.status_code)