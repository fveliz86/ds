import streamlit as st
import pickle
import pandas as pd
import numpy as np
import nltk
from PIL import Image
import random
import pathlib
import sklearn
import tempfile
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")

nltk.download('stopwords', quiet=True, raise_on_error=True, download_dir=tempfile.gettempdir())
nltk.download('punkt', download_dir=tempfile.gettempdir())
nltk.data.path.append(tempfile.gettempdir())

# print(tempfile.gettempdir())

code_dir = pathlib.Path(__file__).parent.resolve()

class Tokenizer:
    def __init__(self, stopwords):
        self.stemmer = SnowballStemmer('spanish')
        self.stopwords = set(stopwords)

    def __call__(self, line):
        tokens = nltk.word_tokenize(line.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.stopwords]
        return [self.stemmer.stem(t) for t in tokens]


# test_prueba='Aquí aay allí debemos hacer todo el esfuerzo por la pandemia. Ver producción juramento'
# test_prueba='Hay oportunidades de buscar un cambio. Gracias por el entusiasmo'
# test_prueba=test.loc[230,'texto']

#Constantes
with open(f'{code_dir}/auxiliar/vectorizer_v2.pickle', 'rb') as handle:
    vectorizer = pickle.load(handle)

with open(f'{code_dir}/auxiliar/modelo_lr.pickle', 'rb') as handle:
    lr = pickle.load(handle)
    
base=pd.read_csv(f'{code_dir}/auxiliar/base_original_test.csv')

rutas_img=[f'{code_dir}/auxiliar/mm.jpg',f'{code_dir}/auxiliar/af.jpg']
rutas_icono=[f'{code_dir}/auxiliar/wrong.jpg',f'{code_dir}/auxiliar/right.jpg']
presidentes=['Mauricio Macri','Alberto Fernández']
with open(f'{code_dir}/auxiliar/nombres_propios_completo.pickle', 'rb') as handle:
        nombres_propios = pickle.load(handle)


frases_random=[]
for i,fila in base.iterrows():
    for parrafo in fila.texto.split('\n'):
        if len(parrafo)>=400:
            lista=parrafo.split(' ')
            frases_random.append([parrafo,fila.clase,i])


if 'discurso' not in st.session_state:
    st.session_state.discurso=''

if 'predecir_clicked' not in st.session_state:
    st.session_state.predecir_clicked = False

if 'iniciar_clicked' not in st.session_state:
    st.session_state.iniciar_clicked = False

if 'pred' not in st.session_state:
    st.session_state.pred = -1

if 'res_genai' not in st.session_state:
    st.session_state.res_genai = -1

if 'res' not in st.session_state:
    st.session_state.res = -1

if 'prob' not in st.session_state:
    st.session_state.prob = -1

if 'acierto' not in st.session_state:
    st.session_state.acierto = -1

if 'aciertos_lista' not in st.session_state:
    st.session_state.aciertos_lista = [0,0,0]

if 'acierto_lista' not in st.session_state:
    st.session_state.acierto_lista = [0,0,0]    

if 'corrida' not in st.session_state:
    st.session_state.corrida = 0

if 'clase_real' not in st.session_state:
    st.session_state.clase_real = -1

if 'clase_real_str' not in st.session_state:
    st.session_state.clase_real_str = -1

if 'iniciar_nombre' not in st.session_state:
    st.session_state.iniciar_nombre='INICIAR JUEGO'

if 'flag_resultados' not in st.session_state:
    st.session_state.flag_resultados=False

if 'llamar_votar' not in st.session_state:
    st.session_state.llamar_votar = False

if 'aux_df' not in st.session_state:
    st.session_state.aux_df = pd.DataFrame()

if 'resultados_figura' not in st.session_state:
    st.session_state.resultados_figura = None

if 'mostrar_votaciones' not in st.session_state:
    st.session_state.mostrar_votaciones = None

if 'cant_partidas' not in st.session_state:
    st.session_state.cant_partidas = 3

if 'fin_juego' not in st.session_state:
    st.session_state.fin_juego=False

if 'mostrar_resultados_final' not in st.session_state:
    st.session_state.mostrar_resultados_final=False

    

import re
from nltk.tokenize.treebank import TreebankWordDetokenizer


def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("¿","?", ".",',', ";", ":", "¡", "!",'"','(',')','-','–','”','“','*'))
#     final = text.replace("¿?.",'')
    return final

def remove_palabras(text,palabras):
    for i in palabras:
        text = text.replace(i,' ')
    return text

def sacar_numeros(text):
    res = ''.join([i for i in text if not i.isdigit()])
    return(res)

def predecir_fn(discurso_texto):

    # palabras=['\n','APLAUSOS','\xa0','…','alberto','fernández','mauricio','macri','ingeniero','Frente de Todos','cristina','gabriela','kirchner','michetti']

    discurso_texto=remove_punctuation(discurso_texto)
    # discurso_texto=remove_palabras(discurso_texto,palabras)
    discurso_texto=sacar_numeros(discurso_texto)
    discurso_texto=discurso_texto.replace(' +',' ')
    discurso_texto=discurso_texto.lower()

    test_prueba_t=vectorizer.transform(pd.Series(discurso_texto))

    feature_names=np.array(vectorizer.get_feature_names_out())
    print(f'{test_prueba_t.shape[1]} vs {len(feature_names)} vs {len(lr.coef_[0])}')

    test_pred = lr.predict_proba(test_prueba_t)
    st.session_state.res=np.argmax(test_pred[0])
    st.session_state.prob=max(test_pred[0])
    st.session_state.acierto=int(st.session_state.res==st.session_state.clase_real)

def click_button():
    st.session_state.predecir_clicked = True
#     predecir()
    

def traer_clicked():
    discurso_prueba=random.choice(frases_random)
    discursos_sin_nombres=discurso_prueba[0]
    for i in nombres_propios:
        discursos_sin_nombres = discursos_sin_nombres.replace(i,'...')
    st.session_state.discurso = discursos_sin_nombres
    st.session_state.clase_real=discurso_prueba[1]


def iniciar_clicked_fn():
    st.session_state.iniciar_clicked = True
    ind_random=random.choice(range(len(frases_random)))
    discurso_prueba=frases_random[ind_random]
    st.session_state.link=base.loc[base.index==discurso_prueba[2],'link'].values[0]
    # discursos_sin_nombres=discurso_prueba[0]
    # for i in nombres_propios:
    #     discursos_sin_nombres = discursos_sin_nombres.replace(i,'...')

    tokens = nltk.word_tokenize(discurso_prueba[0])
    tokens2=[i if i not in nombres_propios else ' ... ' for i in tokens]
    discursos_sin_nombres = TreebankWordDetokenizer().detokenize(tokens2)
    
    st.session_state.discurso_juego = discursos_sin_nombres
    st.session_state.clase_real=discurso_prueba[1]

    if discurso_prueba[1]==0:
        st.session_state.clase_real_str='Mauricio Macri'
    else:
        st.session_state.clase_real_str='Alberto Fernández'
    if st.session_state.corrida==0:
        st.session_state.flag_resultados=False
        st.session_state.aciertos_lista=[0,0,0]
    st.session_state.corrida+=1
    st.session_state.mostrar_votaciones = False


def limpiar_discurso_edi():
    tab3.write('')

def limpiar_discurso_juego():
    tab4.write('')

def genai_func(discurso_juego):

    try:
        client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
        )

        completion = client.chat.completions.create(
        extra_body={},
        model="meta-llama/llama-4-maverick:free",
        messages=
        [
            {
            "role": "user",
            "content": 
            [
                {
                "type": "text",
                "text": f"""¿A qué presidente argentino corresponde el siguiente discurso? Respondeme únicamente con el nombre y apellido del presidente. Las opciones son Alberto Fernández y Mauricio Macri.
                    Discurso: '{discurso_juego}'
                    """
                }        
            ]
            }
        ]
        )
        if 'Mauricio Macri' in completion.choices[0].message.content:
            st.session_state.res_genai=0
        else:
            st.session_state.res_genai=1
        # st.session_state.res_genai=0
    except:
        st.error('No es posible acceder a Meta por haber excedido el límite de consultas.\nVolver a intentar más tarde.', icon="🚨")
    
def opcion_clicked(opcion):
    st.session_state.pred=opcion
    st.session_state.iniciar_nombre='SIGUIENTE RONDA'
    st.session_state.llamar_votar = True  # ← Marcamos que se debe votar
    st.session_state.mostrar_votaciones = False
    votar()

def votar():
    
    st.session_state.flag_resultados=True
    with tab4.status("Corriendo modelos...", expanded=True) as status:
        progreso = status.progress(40, text="Corriendo predicción modelo...")  # inicia en 0%
        # status.write("")
        predecir_fn(st.session_state.discurso_juego)
        progreso.progress(40, text="Corriendo predicción GenAI...") 
        genai_func(st.session_state.discurso_juego)
        progreso.progress(100) 
        status.update(label="Ejecución finalizada", state="complete")

    st.session_state.acierto_lista[0]=int(st.session_state.pred==st.session_state.clase_real)
    st.session_state.acierto_lista[1]=int(st.session_state.res==st.session_state.clase_real)
    st.session_state.acierto_lista[2]=int(st.session_state.res_genai==st.session_state.clase_real)
    
    st.session_state.aciertos_lista[0]+=st.session_state.acierto_lista[0]
    st.session_state.aciertos_lista[1]+=st.session_state.acierto_lista[1]
    st.session_state.aciertos_lista[2]+=st.session_state.acierto_lista[2]

    r=[
    st.session_state.aciertos_lista[0]/st.session_state.corrida*100,
    st.session_state.aciertos_lista[1]/st.session_state.corrida*100,
    st.session_state.aciertos_lista[2]/st.session_state.corrida*100
    ]

    st.session_state.aux_df=pd.DataFrame(np.column_stack([['Humano','Modelo','GenAI'],r]),columns=['Jugador','Puntaje'])
    st.session_state.aux_df['Puntaje']=pd.to_numeric(st.session_state.aux_df['Puntaje']).astype(int)
    st.session_state.aux_df.sort_values(by='Puntaje',ascending=True,inplace=True)


    st.session_state.aux_df['ColorRGBA'] = st.session_state.aux_df['Puntaje'].astype(float).apply(
lambda x: f'rgba(141, 213, 147, {x/100:.2f})'  # Verde claro, opacidad según puntaje
)             
    fig = go.Figure()

    for idx, row in st.session_state.aux_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Puntaje']],
            y=[row['Jugador']],
            orientation='h',
            text=row['Puntaje'],
            textposition='inside',
            marker=dict(color=row['ColorRGBA']),  # Asignamos el color RGBA aquí
        ))

    # Formateo del eje x como porcentaje
    fig.update_layout(
        xaxis_tickformat=".0f%",
        xaxis_title="Puntaje (%)",  # Etiqueta para el eje X
        yaxis_title="Jugador",
        height=200,
        margin=dict(l=40, r=20, t=20, b=30),
        showlegend=False ,
        xaxis=dict(range=[0, 100])
    )
    fig.update_coloraxes(showscale=False)
    # st.session_state.fig.plotly_chart(fig, use_container_width=True)
    st.session_state.resultados_figura=fig



st.header('ANÁLISIS DE DISCURSOS PRESIDENCIALES')

tab1, tab2, tab3, tab4 = st.tabs(["Presentación", "EDA", "Predictor", "Juego"])

#Presentación


#EDA


#Predictor 
discurso_edi = tab3.text_area('Ingresar discurso:','',key='discurso',height=160,on_change=limpiar_discurso_edi)

col1, col2, col3= tab3.columns(3)

traer=col1.button('Traer discurso al azar',on_click=traer_clicked)
generar=col2.button('Generar discurso nuevo')
predecir=col3.button('Predecir', on_click=click_button)
# st.write(st.session_state.predecir_clicked)

if predecir and len(discurso)>0:
    predecir_fn(discurso_edi)
    tab3.write(f"El discurso corresponde a {presidentes[st.session_state.res]}, con un {st.session_state.prob:.1%} de probabilidad.")

    col1, col2, col3= tab3.columns(3)
    with col1:
        st.image(Image.open(rutas_img[st.session_state.res]), width=130,caption='Predicción')

    if st.session_state.clase_real!=999:
        with col2:
            st.image(Image.open(rutas_img[st.session_state.clase_real]), width=130,caption='Real')
        with col3:
            st.image(Image.open(rutas_icono[st.session_state.acierto]), width=200)
    st.session_state.clase_real=999


#Juego    

tab4.subheader(':brain: Zoon politikon :brain:',divider="gray")
tab4.markdown("""
          ¿Estás listo para poner a prueba tus conocimientos sobre el discurso político argentino?.
          Acepta el DESAFÍO y competí tanto contra el modelo entrenado como contra un LLM de GenAI.\n
          Instrucciones:\n
          1. Elegí la cantidad de partidas.\n
          2. Ante cada discurso presidencial, elegí quién creés que lo pronunció.\n
          3. Verás la respuesta del modelo y GenAI. Al finalizar, podrás comparar los resultados.\n\n

          :trophy: ¡Que gane el mejor! :trophy:
          """)

# col1,col2,col3=tab4.columns(3)
if st.session_state.corrida==0 or st.session_state.mostrar_votaciones:
    iniciar=tab4.button(st.session_state.iniciar_nombre,on_click=iniciar_clicked_fn)

if st.session_state.corrida==0:
    # rondas_label=st.write("Cantidad de rondas")
    opcion=tab4.selectbox("Cantidad de rondas",[3,5,10])
    st.session_state.cant_partidas=opcion


if st.session_state.corrida>0:
    ronda=tab4.markdown(f'## :game_die: **Ronda N°{st.session_state.corrida} de {st.session_state.cant_partidas}**')
    # tab4.write(st.session_state.cant_partidas)


if st.session_state.iniciar_clicked:
    #jugador, modelo, meta, real
    # status_container = tab4.container()
    # with status_container.status('Discurso:',expanded=True) as tablero:
    
    if st.session_state.flag_resultados:
    # if "fig" not in st.session_state:
        tab4.subheader(':bar_chart: Resultados')
        # st.session_state.fig = tab4.empty() 
        tab4.plotly_chart(st.session_state.resultados_figura, use_container_width=True)
        tab4.divider()

        if st.session_state.fin_juego==True:
            st.session_state.iniciar_nombre='REINICIAR'
            st.session_state.fin_juego=False
            st.session_state.mostrar_resultados_final=True
            st.session_state.corrida=0
            st.rerun()

    if st.session_state.mostrar_resultados_final:
        # tab4.header('final 2!')
        st.session_state.aux_df['rank']=st.session_state.aux_df['Puntaje'].rank(ascending=False,method='min')
        ranking=st.session_state.aux_df.loc[st.session_state.aux_df['Jugador']=='Humano','rank'].values[0]
        puntaje=st.session_state.aux_df.loc[st.session_state.aux_df['Jugador']=='Humano','Puntaje'].values[0]
        mismo_puesto=list(st.session_state.aux_df.loc[(st.session_state.aux_df['rank']==ranking)&(st.session_state.aux_df['Jugador']!='Humano'),'Jugador'].values)

        if ranking==1:
            imagen_res=f'{code_dir}/auxiliar/1er puesto.png'
            texto=f'¡Felicitaciones! Haz logrado el 1er puesto con un {puntaje}% de aciertos. Eres un auténtico Maquiavelo del análisis político.'
        elif ranking==2:
            imagen_res=f'{code_dir}/auxiliar/2do puesto.png'
            texto=f'¡Muy bien! Haz logrado el 2do puesto con un {puntaje}% de aciertos. No sos el más agudo analista político, pero estás bien encaminado.'
        else:
            imagen_res=f'{code_dir}/auxiliar/3er puesto.png'
            texto=f'Haz logrado el 3er puesto con un {puntaje}% de aciertos. Fuiste derrotado tanto por un modelo de análisis de textos como por un LLM de GenAI.\n¡No te rindas y volvé a participar!'

        if (ranking<3) and (len(mismo_puesto))>0:
            texto+=f'\nCompartiste el puesto con {" y ".join(mismo_puesto)}.'

        col1, col2, col3= tab4.columns([0.33,0.33,0.33])
        col2.image(Image.open(imagen_res), width=400,caption=texto)
        # col2.markdown(texto)
        st.session_state.mostrar_resultados_final=False
     
    if st.session_state.corrida<=st.session_state.cant_partidas:

        col1, col2= tab4.columns([0.85,0.15])
        col1.subheader(':newspaper: Discurso')
        discurso_juego = col1.markdown(st.session_state.discurso_juego)

        if not st.session_state.mostrar_votaciones:
            col2.subheader(':pencil2: Opciones')
            mm_boton=col2.button('Mauricio Macri', on_click=lambda: opcion_clicked(0))
            af_boton=col2.button('Alberto Fernández', on_click=lambda: opcion_clicked(1))

        if st.session_state.llamar_votar:
            # votar()  # ← Ahora sí la función se ejecuta de forma normal
            st.session_state.llamar_votar = False  # ← Limpiamos el flag
            st.session_state.mostrar_votaciones = True
            if st.session_state.corrida==st.session_state.cant_partidas:
                st.session_state.fin_juego=True
                # col_final.header('final!')
            st.rerun()

        if st.session_state.mostrar_votaciones:
            col1.write(f'Ref.: {st.session_state.link}')
            tab4.divider()

            col1,col2=tab4.columns([0.75,0.25])
            col1.subheader(':memo: Votaciones')
            col11,col11b,dummy,col12,col12b,dummy,col13,col13b,dummy= col1.columns([0.13,0.05,0.15,0.13,0.05,0.15,0.13,0.05,0.15])

            col11.image(Image.open(rutas_img[st.session_state.pred]), width=130,caption='Jugador')
            col11b.image(Image.open(rutas_icono[st.session_state.acierto_lista[0]]), width=130)
            
            col12.image(Image.open(rutas_img[st.session_state.res]), width=130,caption='Modelo')
            col12b.image(Image.open(rutas_icono[st.session_state.acierto_lista[1]]), width=130)

            col13.image(Image.open(rutas_img[st.session_state.res_genai]), width=130,caption='GenAI')
            col13b.image(Image.open(rutas_icono[st.session_state.acierto_lista[2]]), width=130)
            
            col2.subheader(':dart: Respuesta correcta')
            col2.image(Image.open(rutas_img[st.session_state.clase_real]), width=130, caption='Real')


# if st.session_state.pred!=-1:
#     votar()

