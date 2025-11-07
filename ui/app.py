# ui/app.py
# Streamlit app - 3 carruseles infinitos + CSV -> OpenAI descriptions
# Requisitos: streamlit, requests, pandas, openai
# Keys: .streamlit/secrets.toml or .env for OPENAI_API_KEY and UNSPLASH_ACCESS_KEY

import streamlit as st
import requests
import pandas as pd
import os
import time
import random
import math
import openai

st.set_page_config(layout="wide", page_title="Turismo Lima IA - 3 Carruseles", page_icon="🗺️")

# -------------------
# Config / keys
# -------------------
UNSPLASH_KEY = st.secrets.get("UNSPLASH_ACCESS_KEY", os.getenv("UNSPLASH_ACCESS_KEY"))
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not UNSPLASH_KEY:
    st.error("Falta UNSPLASH_ACCESS_KEY en st.secrets o en las variables de entorno.")
    st.stop()
if not OPENAI_KEY:
    st.warning("OPENAI_API_KEY no encontrada. Las descripciones IA no funcionarán hasta agregar la key.")
# Configure OpenAI client if key exists
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# -------------------
# Settings
# -------------------
IMAGES_TOTAL = 100
SLEEP_SECONDS = 0.35  # tiempo entre frames (ajusta para velocidad)
UNSPLASH_QUERY = "Lima Peru tourism travel landmarks"  # búsqueda amplia para obtener imágenes variadas
CSV_PATH = "/mnt/data/lugares_lima.csv"  # ajusta si es necesario

# -------------------
# Helpers
# -------------------
@st.cache_data(ttl=86400)
def load_csv(path):
    df = pd.read_csv(path)
    return df

@st.cache_data(ttl=86400)
def fetch_unsplash_images(query=UNSPLASH_QUERY, total=IMAGES_TOTAL):
    images = []
    url = "https://api.unsplash.com/search/photos"
    per_page = 30
    pages = math.ceil(total / per_page)
    for page in range(1, pages + 1):
        params = {
            "query": query,
            "page": page,
            "per_page": per_page,
            "client_id": UNSPLASH_KEY
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            # si hay error devolvemos lo que tengamos (evita crash total)
            st.error(f"Error Unsplash {r.status_code}: {r.text[:200]}")
            break
        data = r.json().get("results", [])
        for item in data:
            images.append({
                "url": item["urls"].get("regular"),
                "alt": item.get("alt_description") or "",
                "author": item.get("user", {}).get("name", "")
            })
            if len(images) >= total:
                return images[:total]
    return images[:total]

def build_pairs(images, df, total=IMAGES_TOTAL):
    # sample filas del CSV (sin replacement si hay suficientes)
    nrows = len(df)
    if nrows >= total:
        sampled = df.sample(n=total, random_state=42).reset_index(drop=True)
    else:
        # repetir si no hay suficientes filas
        sampled = pd.concat([df] * (total // nrows + 1), ignore_index=True).sample(n=total, random_state=42).reset_index(drop=True)
    pairs = []
    for i in range(len(images)):
        row = sampled.loc[i]
        pairs.append({
            "image": images[i],
            "region": str(row.get("REGIÓN", "")),
            "provincia": str(row.get("PROVINCIA", "")),
            "distrito": str(row.get("DISTRITO", "")),
            "nombre": str(row.get("NOMBRE DEL RECURSO", "")),
            "subtipo": str(row.get("SUB TIPO CATEGORÍA", "")),
            "url_recurso": str(row.get("URL", "")),
            "lat": str(row.get("LATITUD", "")),
            "lon": str(row.get("LONGITUD", "")),
        })
    return pairs

def openai_generate_description(data_row):
    """
    data_row: dict con keys 'nombre','distrito','region','provincia','subtipo','lat','lon','url_recurso'
    Retorna texto (string) o None si falla
    """
    if not OPENAI_KEY:
        return "OpenAI API key no configurada. Agrega OPENAI_API_KEY para generar la descripción."
    prompt = f"""
Eres un guía turístico entusiasta y amigable que escribe descripciones cortas y persuasivas en español.
Genera un párrafo atractivo (máx 120 palabras) que motive a visitar el lugar con esta información:
Nombre: {data_row['nombre']}
Distrito: {data_row['distrito']}
Provincia: {data_row['provincia']}
Región: {data_row['region']}
Tipo: {data_row['subtipo']}
Coordenadas: lat {data_row['lat']} lon {data_row['lon']}
URL: {data_row['url_recurso']}

Incluye:
- 1 frase de apertura emotiva.
- 1 detalle atractivo (basado en el tipo o distrito).
- 1 llamada a la acción final ("visítalo", "no te lo pierdas", etc).
Escribe en estilo conversacional y amigable, adecuado para una tarjeta de recomendación turística.
"""
    try:
        # Usamos ChatCompletion o completions según tu cliente openai
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # cambia si prefieres otro
            messages=[
                {"role": "system", "content": "Eres un asistente que escribe descripciones turísticas en español."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.85,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return text
    except Exception as e:
        return f"(Error generando texto con OpenAI: {e})"

# -------------------
# Load data & prepare
# -------------------
st.title("🗺️ Guías Turísticas IA - 3 Carruseles Infinitos")
st.caption("Haz click en 'Seleccionar' bajo una imagen para recibir una recomendación personalizada (usando la info del CSV y OpenAI).")

# Carga CSV y fotos (cacheadas)
try:
    df = load_csv(CSV_PATH)
except Exception as e:
    st.error(f"No se pudo cargar CSV en {CSV_PATH}: {e}")
    st.stop()

images = fetch_unsplash_images()
if len(images) < IMAGES_TOTAL:
    st.warning(f"Se obtuvieron {len(images)} imágenes desde Unsplash. La app funcionará con las que haya.")

pairs = build_pairs(images, df, total=len(images))

# -------------------
# Session state init
# -------------------
if "pos1" not in st.session_state:
    st.session_state.pos1 = 0
if "pos2" not in st.session_state:
    st.session_state.pos2 = 0
if "pos3" not in st.session_state:
    st.session_state.pos3 = 0
if "selected_description" not in st.session_state:
    st.session_state.selected_description = None
if "selected_meta" not in st.session_state:
    st.session_state.selected_meta = None

chunk = math.ceil(len(pairs) / 3)
car1 = pairs[0:chunk]
car2 = pairs[chunk:chunk*2]
car3 = pairs[chunk*2:chunk*3]  # resta

# Ensure equal-length slices (allow shorter last)
len1, len2, len3 = len(car1), len(car2), len(car3)
if len1 == 0:
    st.error("No hay imágenes para mostrar.")
    st.stop()

# -------------------
# Render 3 carousels
# -------------------
cols = st.columns(3)

def render_carousel(column, car, pos_key, direction=1, car_index=1):
    """Renderiza una sola imagen del carrusel con botón seleccionar."""
    idx = st.session_state.get(pos_key, 0) % len(car)
    item = car[idx]
    with column:
        st.image(item["image"]["url"], caption=f"{item['nombre']} — {item['distrito']}", use_column_width=True)
        st.markdown(f"**Palabra clave:** {item['nombre']}")
        if st.button(f"Seleccionar (carrusel {car_index})", key=f"select_{car_index}"):
            # Cuando el usuario selecciona, generar la descripcion y guardarla en session_state
            st.session_state.selected_meta = item
            # Generar con OpenAI (puede tardar)
            with st.spinner("Generando recomendación con OpenAI..."):
                desc = openai_generate_description(item)
            st.session_state.selected_description = desc

# Render 3 carousels with opposite directions
render_carousel(cols[0], car1, "pos1", direction=1, car_index=1)
render_carousel(cols[1], car2, "pos2", direction=-1, car_index=2)
render_carousel(cols[2], car3, "pos3", direction=1, car_index=3)

# -------------------
# Autoplay logic: update positions then rerun
# -------------------
# Update positions (wrap-around)
# Advance pos1 and pos3 forward; pos2 backward (opposite)
st.session_state.pos1 = (st.session_state.pos1 + 1) % (len(car1) or 1)
st.session_state.pos2 = (st.session_state.pos2 - 1) % (len(car2) or 1)
st.session_state.pos3 = (st.session_state.pos3 + 1) % (len(car3) or 1)

# Display generated recommendation if exists
st.markdown("---")
st.subheader("🔔 Recomendación seleccionada")
if st.session_state.selected_meta:
    meta = st.session_state.selected_meta
    st.markdown(f"**Lugar:** {meta['nombre']} — {meta['distrito']}, {meta['provincia']}, {meta['region']}")
    st.markdown(f"**Tipo:** {meta['subtipo']}  •  **Coordenadas:** {meta['lat']} , {meta['lon']}")
    if st.session_state.selected_description:
        st.markdown("**Descripción IA:**")
        st.info(st.session_state.selected_description)
    else:
        st.info("Descripción no generada aún. Haz click en 'Seleccionar' para pedir la descripción a OpenAI.")
else:
    st.info("Haz click en 'Seleccionar' debajo de cualquier imagen para obtener una recomendación.")

# Small footer + credits
st.markdown("---")
st.caption("Nota: Las imágenes son de Unsplash (hotlinking) y la descripción es generada por OpenAI usando datos del CSV.")

# Sleep & rerun to create autoplay effect
time.sleep(SLEEP_SECONDS)
st.experimental_rerun()
