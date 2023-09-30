import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath


st.title('Hayvonlarni klassifikatsiya qiluvchi model')
st.write('Assalomu alaykum bizning loyiha sizga rasmni(Ayiq, Baliq, Qush) tanish qobilyatiga ega.‼️Diqaat siz unga boshqa turdagi rasmlarni yuklamang')
file = st.file_uploader('Rasm yuklash', type=['png','jpg','gif'])


if file:
    st.image(file)
    #PIL covert
    img = PILImage.create(file)

    # #model

    model = load_learner('animel_model.pkl')

    # prediction
    pred , pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
    
    