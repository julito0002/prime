import os
import time
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

load_dotenv()
st.set_page_config(page_title="Lomas - Módulo 3 (Imágenes)", layout="wide")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def fake_retriever(query, k=3):
    time.sleep(0.6)
    resultados = [
        f"Resultado 1: Información general sobre '{query}'.",
        f"Resultado 2: Detalle turístico sobre '{query}'.",
        f"Resultado 3: Consejo útil sobre '{query}'.",
    ]
    return resultados[:k]

st.title("Lomas — Módulo 3: Interfaz Streamlit (Galería)")
st.write("App con búsqueda, 3 sliders y galería de imágenes (30 imágenes por renglón).")

# Búsqueda y selectbox de ejemplo
sample_queries = [
    "Mejores playas Lima",
    "Dónde comer en Miraflores",
    "Rutas de senderismo cerca de Lima",
    "Museos gratuitos en Lima"
]
choice = st.selectbox("Consulta de ejemplo", options=sample_queries)
query = st.text_input("Escribe tu consulta", value=choice)

# Sliders simples (3)
st.subheader("Ajustes visuales (3 sliders)")
cols = st.columns(3)
sliders = []
for i, col in enumerate(cols):
    with col:
        v = st.slider(f"Slider {i+1}", min_value=0, max_value=100, value=50, key=f"s{i+1}")
        sliders.append(v)

# Botón buscar
buscar = st.button("Buscar")
if buscar:
    st.info(f"Buscando resultados para: {query}")
    with st.spinner("Consultando..."):
        resultados = fake_retriever(query)
        time.sleep(0.5)
    st.success("Búsqueda completada.")
    for i, texto in enumerate(resultados, start=1):
        st.markdown(f"**Resultado {i}:** {texto}")

st.markdown("---")

# Galería: 30 imágenes por renglón usando picsum.photos como placeholders
st.subheader("Galería de imágenes (30 por renglón)")
# Generamos 30 URLs de ejemplo (puedes reemplazarlas por URLs reales o rutas locales)
image_ids = list(range(10, 40))  # 30 ids: 10..39
image_urls = [f"https://picsum.photos/id/{i}/400/260" for i in image_ids]

# Mostrar las 30 imágenes en una fila con scroll horizontal
container = st.container()
with container:
    st.write("Desliza horizontalmente para ver todas las imágenes:")
    # Usamos HTML/CSS para crear un div con scroll horizontal
    html = '<div style="display:flex; overflow-x:auto; gap:8px; padding:8px;">'
    for idx, url in enumerate(image_urls, start=1):
        html += f'''
        <div style="flex: 0 0 auto; width: 240px; text-align:center;">
            <img src="{url}" alt="img{idx}" style="width:100%; height:auto; border-radius:8px; border:1px solid #ddd;" />
            <div style="font-size:12px; margin-top:4px;">Img {idx}</div>
        </div>'''
    html += "</div>"
    components.html(html, height=320, scrolling=True)

st.markdown("---")
st.markdown("<small>Incluye imágenes de ejemplo (picsum.photos). Reemplaza `image_urls` con URLs reales o rutas locales 'images/...' si tienes tus fotos.</small>", unsafe_allow_html=True)

