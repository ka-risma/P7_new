import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
import pickle
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


########################
# Lecture des fichiers #
########################
data_test_std = pd.read_csv('Data/P7_data_test_20features_importance_std_sample.csv', sep=",")

#################################################
#     Lecture de l'information d'un client      #
#################################################

liste_clients=list(data_test_interprete['SK_ID_CURR'].values)

seuil = 0.52

# Selection d'un client
ID_client = st.selectbox("Merci de saisir l'identifiant du client:", (liste_clients))

st.text("")


 
    #################################################
    # Lecture du modèle de prédiction et des scores #
    #################################################
# Loading model to compare the results
model_LGBM = pickle.load(open('Data/model_complete.pkl','rb'))
    
# Score client    
X = data_test_std[data_test_std.SK_ID_CURR==int(ID_client)]
X = X.drop(['SK_ID_CURR'], axis=1)
probability_default_payment = model_LGBM.predict_proba(X)[:, 1]
score_value = round(probability_default_payment[0]*100, 2) 
if probability_default_payment >= seuil:
    prediction = "Prêt NON Accordé"
else:
    prediction = "Prêt Accordé" 

# Affichage du Score client 
# Titre 1
st.markdown("""<h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Score du client: </h1>
            """, unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)
with col2:
    original_title = '<p style="font-size: 20px;text-align: center;"> <u>Probabilité d\'être en défaut de paiement : </u> </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    original_title = '<p style="font-family:Courier; color:BROWN; font-size:50px; text-align: center;">{}%</p>'.format((probability_default_payment[0]*100).round(2))
    st.markdown(original_title, unsafe_allow_html=True)

    original_title = '<p style="font-size: 20px;text-align: center;"> <u>Conclusion : </u> </p>'
    st.markdown(original_title, unsafe_allow_html=True)

    if prediction == "Prêt Accordé":
        original_title = '<p style="font-family:Courier; color:GREEN; font-size:70px; text-align: center;">{}</p>'.format(prediction)
        st.markdown(original_title, unsafe_allow_html=True)
    else :
        original_title = '<p style="font-family:Courier; color:red; font-size:70px; text-align: center;">{}</p>'.format(prediction)
        st.markdown(original_title, unsafe_allow_html=True)    
    

    

    
