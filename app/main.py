import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_card import card
from torchvision import transforms
from PIL import Image
from model import *
import plotly.express as px
import torch as th
import numpy as np
import pandas as pd

transform = transforms.Compose([
    transforms.Resize(size=(28,28)),
    transforms.ToTensor()
])

st.set_page_config(
    page_title='Guess Numbers',
    layout='centered'
)

st.markdown(
    """
    <style>
    /* Fondo general */
    .stMain {
        margin: 0;
        padding: 0;
        height: 100%;
        background-size: 100px 100px;
        background-image: 
            linear-gradient(hsl(0, 0%, 39%) 1px, transparent 1px), 
            linear-gradient(to right, transparent 99%, hsl(0, 0%, 39%) 100%);
        mask: radial-gradient(100% 100% at 50% 50%, hsl(0, 0%, 0%, 1), hsl(0, 0%, 0%, 0));
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stButton > button {
        border: none; 
        border-radius: 4px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:active {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:disabled {
        background-color: #9e9e9e;
        cursor: not-allowed;
        box-shadow: none;
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(98, 0, 234, 0.5);
    }
    
    .stAppHeader {
        backdrop-filter: blur(8px) saturate(180%);
        -webkit-backdrop-filter: blur(8px) saturate(180%);
        background-color: rgba(17, 25, 40, 0.75);
        border-radius: 0;
        border: 1px solid rgba(255, 255, 255, 0.125);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stSelectbox select {
        border: 2px solid #6200ea;
        border-radius: 4px;
        padding: 10px 15px;
        font-size: 16px;
        color: #333;
        width: 100%;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .stSelectbox select:focus {
        border-color: #3700b3;
        outline: none;
        box-shadow: 0 0 5px rgba(98, 0, 234, 0.5);
    }

    .stSelectbox option {
        padding: 10px;
        font-size: 16px;
        color: #333;
        transition: background-color 0.2s ease;
    }

    .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Draw Numbers')

with st.sidebar:
    drawing_mode = st.selectbox(
        "Draw Mode:",
        ("Freedraw", "Transform"),
        help="Selecciona el modo de dibujo."
    )
    stroke_width = st.slider("Grosor del trazo:", 1, 25, 3)
    stroke_color = st.color_picker("Color del trazo:", "#000000")
    background_color = st.color_picker("Color de fondo:", "#FFFFFF")
    realtime_update = st.checkbox("Actualización en tiempo real", True)

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=background_color,
    height=280,
    width=280,
    drawing_mode=drawing_mode.lower(),
    key="canvas",
    update_streamlit=realtime_update,
)

if canvas_result.image_data is not None:
    st.write("Previsualización del dibujo:")
    img = Image.fromarray(np.array(canvas_result.image_data, dtype=np.uint8))
    transform_img = transform(img)
    transform_img_numpy = transform_img.numpy()

    res = card(
        title="Selecciona el número que representa este dibujo",
        text="Elige un número de la lista para identificar el dibujo",
        styles={
            'card':{
                'width':'100%',
                'border-radius':'8px',
                'box-shadow':'0 4px 6px rgba(0, 0, 0, 0.1)'
            }
        }
    )

    label = st.selectbox("Selecciona el número:", options=list(CLASSES_MAP.keys()))

    for id, img in enumerate(transform_img_numpy):
        st.image(img, caption=f"Imagen: {id} \nForma: {img.shape}")
        st.write(f'Etiqueta: {label}')

    res_pred = card(
        title="Predicción",
        text="Resultado de la predicción del número",
        styles={
            'card':{
                'width':'100%',
                'border-radius':'8px',
                'box-shadow':'0 4px 6px rgba(0, 0, 0, 0.1)'
            }
        }
    )

    if st.button('Ver predicción'):
        guess_number: GuessNumberModelI = th.load(MODEL_SAVE)
        guess_number.eval()

        df = guess_number.stats.to_pandas()

        df_long = pd.melt(df, id_vars=["epochs"], value_vars=["train_loss_list", "test_loss_list", "train_acc_list", "test_acc_list"],
                          var_name="metric", value_name="value")

        # Crear gráfico de pérdida (train_loss vs test_loss)
        fig_loss = px.line(df, x="epochs", y=["train_loss_list", "test_loss_list"], title="Pérdida de entrenamiento y prueba")
        fig_loss.update_layout(xaxis_title="Época", yaxis_title="Pérdida")

        # Crear gráfico de precisión (train_acc vs test_acc)
        fig_acc = px.line(df, x="epochs", y=["train_acc_list", "test_acc_list"], title="Precisión de entrenamiento y prueba")
        fig_acc.update_layout(xaxis_title="Época", yaxis_title="Precisión (%)")

        # Mostrar los gráficos en Streamlit
        st.plotly_chart(fig_loss)
        st.plotly_chart(fig_acc)
        with th.inference_mode():
            logits = guess_number(transform_img.unsqueeze(dim=1))
            probs = th.softmax(logits, dim=1)

            for i in range(len(probs)):
                predicted_label = th.argmax(probs, dim=1)[i].item()
                predicted_prob = probs[0, predicted_label].item()

                st.write(f'Predicción: {predicted_label}, Probabilidad: {predicted_prob:.2f}')

            st.write(CLASSES_MAP)

        st.title('DataFrame de las estadisticas')
        st.write(df)

        lb = th.LongTensor([CLASSES_MAP[label],CLASSES_MAP[label],CLASSES_MAP[label],CLASSES_MAP[label]])
        for epoch in range(10):
            guess_number.optimizer.zero_grad()

            logits = guess_number(transform_img.unsqueeze(dim=1))

            loss = guess_number.loss_fn(logits, lb)

            loss.backward()
            guess_number.optimizer.step()

            st.write(f"Época {epoch + 1}, Pérdida: {loss.item():.4f}")

        th.save(guess_number, MODEL_SAVE)

        #2.4224