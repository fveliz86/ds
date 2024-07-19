import streamlit as st
import pickle
import pandas as pd
import nltk
from PIL import Image
import random

class Tokenizer(object):
    def __init__(self,stopwords):
        nltk.download('punkt', quiet=True, raise_on_error=True)
        self.stemmer = nltk.stem.PorterStemmer()
        self.stopwords=stopwords
        
    def _stem(self, token):
#         print(stopwords)
        if (token in self.stopwords):
            return token  # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."
        return self.stemmer.stem(token)
        
    def __call__(self, line):
        tokens = nltk.word_tokenize(line)
        tokens = (self._stem(token) for token in tokens)  # Stemming
        return list(tokens)

# test_prueba='Aquí aay allí debemos hacer todo el esfuerzo por la pandemia. Ver producción juramento'
# test_prueba='Hay oportunidades de buscar un cambio. Gracias por el entusiasmo'
# test_prueba=test.loc[230,'texto']

with open('vectorizer.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

with open('modelo_lr.pickle', 'rb') as handle:
    lr = pickle.load(handle)

    
base_completa=pd.read_csv('base_completa.csv')

frases_random=[]
for discurso,clase in zip(base_completa['texto'],base_completa['clase']):
    for parrafo in discurso.split('\n'):
        if len(parrafo)>300:
            lista=parrafo.split(' ')
            if 'presidente' not in parrafo.lower():
                frases_random.append([parrafo,clase])

if 'clase_real' not in st.session_state:
    st.session_state.clase_real = 999

if 'discurso' not in st.session_state:
    st.session_state.discurso=''

if 'predecir_clicked' not in st.session_state:
    st.session_state.predecir_clicked = False

def click_button():
    st.session_state.predecir_clicked = True
#     predecir()
    
# if 'flag_texto_propio' not in st.session_state:
#     st.session_state.flag_texto_propio = False

def traer_clicked():
    discurso_prueba=random.choice(frases_random)
    st.session_state.discurso = discurso_prueba[0]
    st.session_state.clase_real=discurso_prueba[1]
#     st.session_state.flag_texto_propio = False

def cambiar_flag():
    tab3.write('')

st.header('ANÁLISIS DE DISCURSOS PRESIDENCIALES')

tab1, tab2, tab3 = st.tabs(["Presentación", "EDA", "Predictor"])

# tab1.subheader("A tab with a chart")
# tab1.line_chart(data)

# tab2.subheader("A tab with the data")

# discurso_prueba='la pandemia ha hecho estragos. Hemos lanzado los ATPs para hacer frente al covid'
   
title = tab3.text_area('Ingresar discurso:','',key='discurso',height=160,on_change=cambiar_flag)

col1, col2, col3= tab3.columns(3)

traer=col1.button('Traer discurso al azar',on_click=traer_clicked)
generar=col2.button('Generar discurso nuevo')
predecir=col3.button('Predecir', on_click=click_button)
# st.write(st.session_state.predecir_clicked)


if predecir and len(title)>0:
    test_prueba_t=vectorizer.transform(pd.Series(title))
    test_pred = lr.predict_proba(test_prueba_t)
    if test_pred[0][0]>test_pred[0][1]:
        res=test_pred[0][0]
        image_pred = Image.open('mm.jpg')
        image_no_pred = Image.open('af.jpg')
        presidente='Mauricio Macri'
        if st.session_state.clase_real==0:
            ruta_icono='right.jpg'
        else:                   
            ruta_icono='wrong.jpg'

    else:
        res=test_pred[0][1]
        image_pred = Image.open('af.jpg')
        image_no_pred = Image.open('mm.jpg')
        presidente='Alberto Fernández'
        if st.session_state.clase_real==1:
            ruta_icono='right.jpg'
        else:                   
            ruta_icono='wrong.jpg'


    tab3.write(f"El discurso corresponde a {presidente}, con un {res:.1%} de probabilidad.")

    col1, col2, col3= tab3.columns(3)
    with col1:
        st.image(image_pred, width=130,caption='Predicción')

    if st.session_state.clase_real!=999:
        with col2:
            if ruta_icono=='right.jpg':
                st.image(image_pred, width=130,caption='Real')
            else:
                st.image(image_no_pred, width=130,caption='Real')
        with col3:
            icono = Image.open(ruta_icono)
            st.image(icono, width=200)
    st.session_state.clase_real=999

