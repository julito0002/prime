# ui/app.py
# Streamlit app - 3 carruseles infinitos (10 imgs cada uno) + CSV -> OpenAI recomendación modal
# Requisitos: streamlit, requests, pandas, openai
# Keys: .streamlit/secrets.toml or variables de entorno:
#   UNSPLASH_ACCESS_KEY = "..." 
#   OPENAI_API_KEY = "..."

import streamlit as st
import requests
import pandas as pd
import os
import time
import math
import random

# -------------------------
# Config
# -------------------------
st.set_page_config(layout="wide", page_title="Turismo Lima IA - 3 Carruseles", page_icon="🗺️")

UNSPLASH_KEY = st.secrets.get("UNSPLASH_ACCESS_KEY", os.getenv("UNSPLASH_ACCESS_KEY"))
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

CSV_PATH = "/workspaces/prime/lugares_lima.csv"   # ajusta si tu CSV está en otra ruta
IMAGES_PER_CAROUSEL = 10
TOTAL_IMAGES = IMAGES_PER_CAROUSEL * 3
AUTOPLAY_DELAY = 0.45  # segundos entre frames (ajusta velocidad)

# -------------------------
# Validación keys
# -------------------------
if not UNSPLASH_KEY:
    st.error("Falta UNSPLASH_ACCESS_KEY en st.secrets o variable de entorno. Agrega la key y reinicia.")
    st.stop()

# OPENAI optional but recommended
if not OPENAI_KEY:
    st.warning("OPENAI_API_KEY no encontrada. Las descripciones IA no funcionarán hasta agregar la key.")

# -------------------------
# OpenAI client helper (compatible con versiones nuevas/viejas)
# -------------------------
def get_openai_client():
    """
    Intenta devolver un cliente compatible tanto con openai v3+ (OpenAI class)
    como con la API tradicional (module openai).
    Retorna (client, kind) donde kind in {"new","old","none"}.
    """
    if not OPENAI_KEY:
        return (None, "none")
    try:
        # New-style client
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        return (client, "new")
    except Exception:
        try:
            import openai
            openai.api_key = OPENAI_KEY
            return (openai, "old")
        except Exception:
            return (None, "none")

OPENAI_CLIENT, OPENAI_KIND = get_openai_client()

# -------------------------
# Utilities
# -------------------------
@st.cache_data(ttl=86400)
def load_csv(path):
    df = pd.read_csv(path)
    return df

@st.cache_data(ttl=86400)
def fetch_unsplash_images(query="Peru travel", total=TOTAL_IMAGES):
    """
    Busca imágenes en Unsplash (paginado) y devuelve una lista de dicts:
    {url, alt, author}
    """
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
            st.warning(f"Unsplash returned {r.status_code}. Returning found images ({len(images)}).")
            break
        results = r.json().get("results", [])
        for it in results:
            images.append({
                "url": it["urls"].get("regular"),
                "alt": it.get("alt_description") or "",
                "author": it.get("user", {}).get("name", "")
            })
            if len(images) >= total:
                return images[:total]
    return images[:total]

def sample_lima_keywords(df, n):
    """
    Extrae n palabras-clave de la columna 'NOMBRE DEL RECURSO' pero
    favorece filas que pertenezcan a Lima (PROVINCIA o DISTRITO contiene 'Lima').
    Si no hay suficientes en Lima, completa con sample global.
    Devuelve una lista de strings.
    """
    df = df.fillna("")
    # filas en Lima (diferentes heurísticas)
    mask = df["PROVINCIA"].str.contains("Lima", case=False, na=False) | df["DISTRITO"].str.contains("Lima", case=False, na=False) | df["REGIÓN"].str.contains("Lima", case=False, na=False)
    lima_rows = df[mask]
    keywords = []
    if len(lima_rows) >= n:
        keywords = lima_rows.sample(n=n, random_state=42)["NOMBRE DEL RECURSO"].astype(str).tolist()
    else:
        # tomar las que haya en Lima primero
        keywords = lima_rows["NOMBRE DEL RECURSO"].astype(str).tolist()
        remaining = n - len(keywords)
        others = df[~mask].sample(n=remaining, random_state=42)["NOMBRE DEL RECURSO"].astype(str).tolist()
        keywords += others
    # Normalize/strip
    keywords = [k.strip() or "atractivo turístico" for k in keywords]
    return keywords

def build_pairs(images, keywords):
    """
    Empareja cada imagen con una keyword (palabra clave).
    images: list of dicts
    keywords: list of strings (len == len(images))
    Retorna lista de dicts con image + keyword
    """
    pairs = []
    for i, img in enumerate(images):
        kw = keywords[i] if i < len(keywords) else f"atractivo {i}"
        pairs.append({
            "image": img,
            "keyword": kw
        })
    return pairs

def find_similar_in_lima(df, keyword):
    """
    Busca en el CSV lugares en LIMA relacionados con 'keyword'.
    Estrategia:
      1) Filtrar filas que pertenecen a Lima (PROVINCIA o DISTRITO o REGIÓN)
      2) Intentar match por tokens: si algún token del keyword aparece en 'NOMBRE DEL RECURSO'
      3) Si no hay match, devolver una fila aleatoria en Lima
    Devuelve dict con metadata de la fila seleccionada.
    """
    df = df.fillna("")
    mask = df["PROVINCIA"].str.contains("Lima", case=False, na=False) | df["DISTRITO"].str.contains("Lima", case=False, na=False) | df["REGIÓN"].str.contains("Lima", case=False, na=False)
    df_lima = df[mask].reset_index(drop=True)
    if df_lima.empty:
        return None
    kw_tokens = [t.lower() for t in keyword.split() if len(t) > 2]
    # Try to find best match by token presence in 'NOMBRE DEL RECURSO'
    for token in kw_tokens:
        candidates = df_lima[df_lima["NOMBRE DEL RECURSO"].str.lower().str.contains(token, na=False)]
        if not candidates.empty:
            chosen = candidates.sample(n=1).iloc[0]
            return chosen.to_dict()
    # Fallback: try match by SUB TIPO CATEGORÍA token
    for token in kw_tokens:
        candidates = df_lima[df_lima["SUB TIPO CATEGORÍA"].str.lower().str.contains(token, na=False)]
        if not candidates.empty:
            chosen = candidates.sample(n=1).iloc[0]
            return chosen.to_dict()
    # Final fallback: random in Lima
    chosen = df_lima.sample(n=1).iloc[0]
    return chosen.to_dict()

def generate_openai_description(row_meta, keyword):
    """
    Envía prompt a OpenAI para generar una descripción persuasiva.
    row_meta: dict con columnas del CSV
    keyword: palabra clave/imagen original
    Retorna string (o mensaje de error).
    """
    if OPENAI_KIND == "none" or OPENAI_CLIENT is None:
        return "OpenAI no configurado: agrega OPENAI_API_KEY para generar la descripción."

    # Construir prompt en español
    nombre = row_meta.get("NOMBRE DEL RECURSO", "") or row_meta.get("nombre", "")
    distrito = row_meta.get("DISTRITO", "")
    provincia = row_meta.get("PROVINCIA", "")
    region = row_meta.get("REGIÓN", "")
    subtipo = row_meta.get("SUB TIPO CATEGORÍA", "")
    coords = f"{row_meta.get('LATITUD','')}, {row_meta.get('LONGITUD','')}"
    base_url = row_meta.get("URL", "")

    prompt = f"""
Eres un guía turístico conversacional en español. Crea una recomendación atractiva (2-4 frases, máximo 100 palabras) para motivar a visitar este lugar en Lima.
Usa la información:

- Lugar: {nombre}
- Distrito: {distrito}
- Provincia: {provincia}
- Región: {region}
- Tipo/Categoría: {subtipo}
- Coordenadas: {coords}
- Fuente: {base_url}
- Palabra clave / imagen original: {keyword}

Incluye una frase de apertura emotiva, un detalle que destaque y una llamada a la acción breve (ej. "visítalo", "no te lo pierdas"). Escribe en tono amable y directo.
    """

    try:
        if OPENAI_KIND == "new":
            # nuevo cliente OpenAI()
            resp = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini" if hasattr(OPENAI_CLIENT, "chat") else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente que escribe descripciones turísticas en español."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=220,
                temperature=0.8,
            )
            # new client response path
            text = None
            # respuestas pueden aparecer en different keys; intentar leer robustamente:
            if isinstance(resp, dict) and "choices" in resp:
                text = resp["choices"][0]["message"]["content"].strip()
            else:
                # objeto tipo OpenAIResponse
                try:
                    text = resp.choices[0].message.content.strip()
                except Exception:
                    text = str(resp)
            return text
        else:
            # Cliente "old" openai module
            resp = OPENAI_CLIENT.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente que escribe descripciones turísticas en español."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=220,
                temperature=0.8,
            )
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(Error OpenAI: {e})"

# -------------------------
# App: load assets
# -------------------------
st.title("🗺️ Turismo Lima IA — 3 Carruseles Infinitos")
st.caption("30 imágenes (3 carruseles x 10). Haz click en 'Seleccionar' para recibir una recomendación basada en el CSV y generada por OpenAI.")

# Cargar CSV
try:
    df_all = load_csv(CSV_PATH)
except Exception as e:
    st.error(f"No se pudo cargar CSV en {CSV_PATH}: {e}")
    st.stop()

# Cargar imágenes Unsplash
images = fetch_unsplash_images(query="Peru travel landmarks", total=TOTAL_IMAGES)
if not images:
    st.error("No se obtuvieron imágenes desde Unsplash. Revisa la clave o el servicio.")
    st.stop()

# Preparar keywords: preferentemente extraemos keywords desde CSV centrados en Lima
keywords = sample_lima_keywords(df_all, TOTAL_IMAGES)
pairs = build_pairs(images, keywords)

# Dividir en 3 arrays (10 cada una)
def chunk_list(lst, chunk_size):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

chunks = chunk_list(pairs, IMAGES_PER_CAROUSEL)
# En caso de que la lista no sea divisible, asegurar 3 listas:
while len(chunks) < 3:
    chunks.append([])

car1, car2, car3 = chunks[0], chunks[1], chunks[2]

# -------------------------
# Session state: positions & selection
# -------------------------
if "pos1" not in st.session_state: st.session_state.pos1 = 0
if "pos2" not in st.session_state: st.session_state.pos2 = 0
if "pos3" not in st.session_state: st.session_state.pos3 = 0
if "paused" not in st.session_state: st.session_state.paused = False
if "selected_pair" not in st.session_state: st.session_state.selected_pair = None
if "selected_meta" not in st.session_state: st.session_state.selected_meta = None
if "selected_description" not in st.session_state: st.session_state.selected_description = None

# -------------------------
# Render UI: three columns / carousels
# -------------------------
cols = st.columns(3)

def render_single_carousel(column, car, pos_key, car_idx, direction=1):
    """Render the current item of the carousel and the 'Seleccionar' button."""
    if not car:
        with column:
            st.write("No hay imágenes en este carrusel.")
        return
    idx = st.session_state.get(pos_key, 0) % len(car)
    item = car[idx]
    with column:
        # Mostrar imagen (con ancho relativo)
        st.image(item["image"]["url"], caption=f"{item['keyword']}", use_column_width=True)
        st.markdown(f"**Clave:** {item['keyword']}")
        if st.button(f"Seleccionar (Carrusel {car_idx})", key=f"select_{car_idx}"):
            # pause autoplay when user selects
            st.session_state.paused = True
            st.session_state.selected_pair = item
            # find similar in Lima from CSV
            with st.spinner("Buscando lugar similar en Lima..."):
                meta = find_similar_in_lima(df_all, item["keyword"])
                st.session_state.selected_meta = meta
            # generate description via OpenAI
            with st.spinner("Generando recomendación con OpenAI..."):
                desc = generate_openai_description(st.session_state.selected_meta or {}, item["keyword"])
                st.session_state.selected_description = desc
            # open modal below (after finishing)
            open_modal_with_selection()

def open_modal_with_selection():
    """Muestra modal con la recomendación completa."""
    meta = st.session_state.selected_meta
    desc = st.session_state.selected_description
    pair = st.session_state.selected_pair
    # Use streamlit modal
    with st.modal("🔔 Recomendación turística"):
        st.header("🎯 Recomendación seleccionada")
        if pair:
            st.image(pair["image"]["url"], use_column_width=True)
            st.markdown(f"**Keyword (imagen):** {pair['keyword']}")
        if meta:
            st.markdown(f"**Lugar elegido (CSV):** {meta.get('NOMBRE DEL RECURSO','')}")
            st.markdown(f"**Ubicación:** {meta.get('DISTRITO','')}, {meta.get('PROVINCIA','')}, {meta.get('REGIÓN','')}")
            st.markdown(f"**Tipo:** {meta.get('SUB TIPO CATEGORÍA','')}")
            if meta.get("LATITUD") or meta.get("LONGITUD"):
                st.markdown(f"**Coordenadas:** {meta.get('LATITUD','')} , {meta.get('LONGITUD','')}")
            if meta.get("URL"):
                st.markdown(f"[Fuente]({meta.get('URL')})")
        if desc:
            st.subheader("Descripción generada por IA")
            st.info(desc)
        else:
            st.info("No se generó descripción. Agrega OPENAI_API_KEY o reintenta.")
        # botón para cerrar modal y reanudar autoplay
        if st.button("Cerrar"):
            st.session_state.paused = False
            st.experimental_rerun()

# Render each carousel (center one will go reverse visually by updating pos in opposite direction)
render_single_carousel(cols[0], car1, "pos1", 1, direction=1)
render_single_carousel(cols[1], car2, "pos2", 2, direction=-1)
render_single_carousel(cols[2], car3, "pos3", 3, direction=1)

# -------------------------
# Autoplay: advance positions if not paused
# -------------------------
if not st.session_state.paused:
    # advance pos1 forward, pos2 backward, pos3 forward
    if car1:
        st.session_state.pos1 = (st.session_state.pos1 + 1) % len(car1)
    if car2:
        st.session_state.pos2 = (st.session_state.pos2 - 1) % len(car2)
    if car3:
        st.session_state.pos3 = (st.session_state.pos3 + 1) % len(car3)
    # small delay then rerun to animate
    time.sleep(AUTOPLAY_DELAY)
    st.experimental_rerun()
else:
    # Paused — show instruction
    st.info("Autoplay en pausa mientras ves la recomendación. Cierra la ventana para reanudar.")
