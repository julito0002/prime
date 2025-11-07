# streamlit_config.py - Módulo 3: Configuración Streamlit (dotenv, responsive, cache, ética UI)
# Alineado cronograma: Día 2 dotenv key; Día 3 muestra query; Día 4 loading spinner; Día 5 clics query; Día 6 3 filas; Día 7 teléfono test; Día 8 lags; Día 9 ética UI
import streamlit as st  # UI
#from dotenv import load_dotenv  # Día 2: Key .env
import os  # Para env
import logging  # Debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_streamlit():
    """
    Configuración central de Streamlit: Key dotenv, layout responsive, cache, ética.
    """
    # Día 2: Carga .env (API keys OpenAI/Unsplash)
    #load_dotenv()
    api_openai = os.getenv('OPENAI_API_KEY')
    api_unsplash = os.getenv('UNSPLASH_ACCESS_KEY')
    if not api_openai or not api_unsplash:
        logger.error("Keys missing in .env – agrega OPENAI_API_KEY y UNSPLASH_ACCESS_KEY")
        st.stop()  # Para app si falla

    # Día 7: Config responsive (wide para sliders, mobile OK)
    st.set_page_config(
        page_title="Guías IA Lima - Unsplash Sliders",
        page_icon="🗺️",
        layout="wide",  # Ancho para grid 5 cols
        initial_sidebar_state="expanded"  # Sidebar abierto
    )

    # Día 6: Cache global para Unsplash (24h, para bulk 100 imgs)
    @st.cache_data(ttl=86400)
    def global_cache_func(*args, **kwargs):
        return args, kwargs  # Placeholder – usa en fetch_unsplash_images()

    logger.info("Streamlit configurado: Keys loaded, layout wide, cache 24h")

    # Día 9: Config ética UI (disclaimer privacidad)
    st.sidebar.info("### Nota Ética\nDatos en .env seguros (no compartidos). Recomendaciones basadas en CSV público PromPerú – no inventa info.")

# Día 3: Función para mostrar query (feedback UX)
def show_query_feedback(query):
    if query:
        st.sidebar.success(f"Query activa: '{query}' – Buscando en Lima...")
    else:
        st.sidebar.warning("Ingresa una query para generar recomendaciones.")

# Día 4: Loading Spinner Custom (para generación)
def custom_spinner(text="Procesando RAG + GA..."):
    with st.spinner(text):
        yield
        st.balloons()  # Feedback visual OK

# Día 5: Para clics query (session state)
def handle_click_query(query):
    if query:
        st.session_state.current_query = query
    return st.session_state.get('current_query', '')

# Día 6: Para 3 filas sliders (config cols)
def get_columns_layout(num_cols=4):
    return st.columns(num_cols)

# Día 7: Test teléfono (responsive check – manual)
def test_mobile():
    st.sidebar.button("Test Móvil (F12 en browser)")  # Placeholder – usuario prueba manual

# Día 8: Para lags (config cache)
@st.cache_data(ttl=3600)  # 1h para lags UI
def cache_ui_element(key):
    return key  # Placeholder – usa para df o imgs

# Día 9: Config ética UI (disclaimer)
def ethics_disclaimer():
    st.caption("**Ética**: Recomendaciones basadas en datos públicos. Privacidad: Queries anónimas, no almacenadas.")

# Test (Día 2-9)
if __name__ == "__main__":
    configure_streamlit()
    print("Config test OK: Layout wide, keys loaded")