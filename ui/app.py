# app.py - App Streamlit con Unsplash API para Sliders Infinitos
# Función fetch_unsplash_images(): Bulk load 100 imgs, cache 24h, paginación.
# Instalación: pip install streamlit requests python-dotenv
# Key: .env o st.secrets["UNSPLASH_ACCESS_KEY"]
import streamlit as st
import requests  # Para API
from dotenv import load_dotenv  # Para key
import os
import math  # Para paginación
from streamlit_config import configure_streamlit, show_query_feedback, custom_spinner, ethics_disclaimer
configure_streamlit()  # Llama al inicio
# En button: with custom_spinner("Buscando..."):
# Después: ethics_disclaimer()

load_dotenv()  # Carga .env si usas

# Config API Unsplash (usa st.secrets en Cloud o .env local)
ACCESS_KEY = st.secrets.get("UNSPLASH_ACCESS_KEY", os.getenv("UNSPLASH_ACCESS_KEY"))
if not ACCESS_KEY:
    st.error("Agrega UNSPLASH_ACCESS_KEY en secrets o .env (Client-ID de unsplash.com/developers)")
    st.stop()

st.set_page_config(page_title="Guías Turísticas IA Lima - Unsplash Sliders", page_icon="🗺️", layout="wide")

st.title("🗺️ Guías Turísticas IA Lima - Sliders Infinitos con Unsplash")

# Sidebar para Query (Día 2: Input)
query = st.sidebar.text_input("Busca imágenes turísticas de Lima:", placeholder="Ej: playas Lima, Barranco cultura")

if query:
    # Función fetch_unsplash_images() – Bulk load con paginación y cache
    @st.cache_data(ttl=86400)  # Cache 24h para no repetir requests
    def fetch_unsplash_images(query, total=100):
        images = []
        per_page = 30  # Max per request
        pages = math.ceil(total / per_page)  # 4 pages para 100
        url_base = "https://api.unsplash.com/search/photos"
        
        for page in range(1, pages + 1):
            params = {
                'query': query,  # Búsqueda (e.g., "lima playas")
                'per_page': per_page,
                'page': page,
                'client_id': ACCESS_KEY  # Client-ID en header (Authorization: Client-ID {key})
            }
            response = requests.get(url_base, params=params)
            if response.status_code == 200:
                data = response.json()
                for photo in data['results']:
                    images.append({
                        'url': photo['urls']['regular'],  # URL imagen (hotlinking)
                        'description': photo['alt_description'] or 'Imagen turística de Lima',
                        'author': photo['user']['name']  # Atribución Unsplash
                    })
                st.info(f"Página {page} cargada: {len(data['results'])} imágenes de '{query}'")
            else:
                st.error(f"Error API {response.status_code}: {response.text[:100]}... – Verifica key o límites (50 req/hora)")
                break
            if len(images) >= total:
                break
        
        return images[:total]  # Limita a total

    # Llama función (Día 2: Fetch)
    images = fetch_unsplash_images(query, total=100)
    
    if images:
        # Grid de Miniaturas (Día 6: Visual)
        st.subheader(f"Grid de {len(images)} Imágenes de '{query}' (Unsplash)")
        cols = st.columns(5)  # 5 columnas para grid
        for i, col in enumerate(cols * math.ceil(len(images) / 5)):
            if i < len(images):
                img = images[i]
                with col:
                    # Miniatura (50x50) con caption atribución
                    st.image(img['url'], caption=f"{img['description']} by {img['author']}", use_column_width=True, width=100)
        
        # st.slider para Navegar (Snippet clave: 0-99 → imagen selected)
        selected_idx = st.slider("Navega en Slider Infinito (0-99):", 0, len(images)-1, 0)
        selected_img = images[selected_idx]
        st.subheader(f"Imagen Seleccionada #{selected_idx + 1}")
        st.image(selected_img['url'], caption=f"{selected_img['description']} – Autor: {selected_img['author']}", use_column_width=True)
        st.caption("Atribución: Unsplash – Imágenes royalty-free para uso educativo.")
    else:
        st.warning("No imágenes encontradas – prueba otra query o verifica key.")

# Footer (Día 8: Créditos)
st.markdown("---")
st.markdown("""
### Créditos
- API Unsplash para imágenes dinámicas en bulk.
- Tesis UPBJ IA 2025 – Sliders infinitos para UX turística en Lima.
""")

# Test (Día 1-10)
if __name__ == "__main__":
    print("Test app: Unsplash fetch OK – Run with streamlit for full UI.")