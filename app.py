import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from scipy.stats import mode
import pandas as pd
import numpy as np
import os

# Paths de los pesos de los modelos usados
tokenizer_dir = 'code/app/tokenizer'
model_dir = 'code/app/model'

tokenizer_dir = os.path.join(tokenizer_dir)
model_dir = os.path.join(model_dir)
# Scalers
scaler_name_dir = 'code/app/scaler/scaler_names.joblib'
scaler_nit_dir = 'code/app/scaler/scaler_nits.joblib'
# PCA
pca_name_dir = 'code/app/pca/names_pca.joblib'
pca_nit_dir = 'code/app/pca/nits_pca.joblib'
pca_entities_dir = 'code/app/pca/entidades_pca.joblib'
# Trees
names_trees_dir = 'code/app/trees/names'
nits_trees_dir = 'code/app/trees/nits'
entities_trees_dir = 'code/app/trees/entities'

# Datasets que se mostrarán en pantalla como sugerencias
entidades_por_nombre = pd.read_csv('code/app/entidades_agrupadas_por_nombres.csv')
entidades_por_nit = pd.read_csv('code/app/entidades_agrupadas_por_nits.csv')
entidades_por_nombre_y_nit = pd.read_csv('code/app/entidades_agrupadas_por_nombres_y_nits.csv')

def load_all_models(model_dir):
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.joblib'):
            model_path = os.path.join(model_dir, filename)
            model = joblib.load(model_path)
            models.append(model)
    return models

def predict_with_trees(new_data, decision_trees):
    predictions = []
    for tree in decision_trees:
        pred = tree.predict(new_data)
        if pred[0] != -1:
            predictions.append(pred[0])
    return mode(predictions).mode

# Cargar el tokenizador y el modelo
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    # Scalers
    scaler_names = joblib.load(scaler_name_dir)
    scaler_nits = joblib.load(scaler_nit_dir)
    # PCA
    pca_names = joblib.load(pca_name_dir)
    pca_nits = joblib.load(pca_nit_dir)
    pca_entities = joblib.load(pca_entities_dir)
    # Trees
    names_trees = load_all_models(names_trees_dir)
    nits_trees = load_all_models(nits_trees_dir)
    entities_trees = load_all_models(entities_trees_dir)
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")

# Título de la aplicación
st.title('Buscador de entidades')

name_text = st.text_input('Ingresa el nombre de la entidad que vas a registrar:')
nit_text = st.text_input('Ingresa el NIT de la entidad que vas a registrar:')

# Realizar predicción y mostrar resultados
if st.button('Validar registros similares'):
    if name_text and not nit_text:
        st.subheader('Posibles entidades que concuerdan con tú descripción:')
        clase = predict_with_trees(pca_names.transform(scaler_names.transform(model(**tokenizer([name_text], padding=True, truncation=True, return_tensors='pt')).logits.detach().numpy())), names_trees)
        st.dataframe(entidades_por_nombre[entidades_por_nombre['grupo'] == clase][['NOMBRE', 'NIT']])
        st.text('En caso de que no encuentres el registro que deseas registrar,\nte recomendamos proporcionar un nombre más detallado o el nit.')
    elif nit_text and not name_text:
        st.subheader('Posibles entidades que concuerdan con aquí descripción:')
        clase = predict_with_trees(pca_nits.transform(scaler_nits.transform(model(**tokenizer([nit_text], padding=True, truncation=True, return_tensors='pt')).logits.detach().numpy())), nits_trees)
        st.dataframe(entidades_por_nit[entidades_por_nit['grupo'] == clase][['NOMBRE', 'NIT']])
        st.text('En caso de que no encuentres el registro que deseas registrar,\nte recomendamos proporcionar el nombre.')
    elif name_text and nit_text:
        st.subheader('Posibles entidades que concuerdan con ambos descripciones:')
        clase = predict_with_trees(pca_entities.transform(np.concatenate((scaler_names.transform(model(**tokenizer([name_text], padding=True, truncation=True, return_tensors='pt')).logits.detach().numpy()), 
                                                                         scaler_nits.transform(model(**tokenizer([nit_text], padding=True, truncation=True, return_tensors='pt')).logits.detach().numpy())), axis=1)), 
                                                                         entities_trees)
        st.dataframe(entidades_por_nombre_y_nit[entidades_por_nombre_y_nit['grupo'] == clase][['NOMBRE', 'NIT']])
        st.text('En caso de que no encuentres el registro que deseas registrar,\nte recomendamos proporcionar un nombre más detallado o registrar \nel nuevo registro.')
    else:
        st.error('Por favor, ingrese un texto válido.')
