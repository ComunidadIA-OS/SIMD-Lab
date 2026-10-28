import os
import streamlit as st
from src.api.main_controller import fetch_analyzed_documents, get_lexical_profile
from src.explainability.scoring_model import CorporateGenderScorer
from frontend.components.search_sidebar import render_sidebar
from frontend.components.tab_esg_dashboard import render_esg_dashboard
from frontend.components.tab_xai_shap_lime import render_xai_tab
from frontend.components.tab_ai_act_fairness import render_ai_act_tab

st.set_page_config(page_title="SIMD Lab Gender Auditor", layout="wide")


def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Montserrat:wght@400;500;600&display=swap');

            /* Fondo general y tamaño base de texto reducido */
            html, body, [class*="css"] {
                font-size: 14px;
            }

            .stApp {
                background-color: #F4F6F9;
                color: #1E293B;
            }

            /* Barra lateral */
            section[data-testid="stSidebar"] {
                background-color: #EAECEF !important;
                border-right: 1px solid #CBD5E1;
            }

            /* Tipografía general y etiquetas más compactas */
            body, p, li, div.stMarkdown, .stCaption, label, [data-testid="stWidgetLabel"] p {
                font-family: 'Montserrat', sans-serif !important;
                font-size: 0.88rem !important;
            }

            /* Títulos proporcionales y más contenidos */
            h1 { font-size: 1.65rem !important; }
            h2 { font-size: 1.35rem !important; }
            h3 { font-size: 1.15rem !important; }
            h4, h5, h6 { font-size: 1.00rem !important; }

            h1, h2, h3, h4, h5, h6,
            .stDataFrame, table {
                font-family: 'Merriweather', Georgia, serif !important;
                color: #0F172A !important;
            }

            /* Métricas: reducción del tamaño del número y etiqueta */
            div[data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                padding: 10px 14px;
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            [data-testid="stMetricValue"] {
                font-family: 'Merriweather', Georgia, serif !important;
                font-size: 1.45rem !important;
                color: #0F172A !important;
            }
            [data-testid="stMetricLabel"] {
                font-family: 'Montserrat', sans-serif !important;
                font-size: 0.8rem !important;
                color: #64748B !important;
            }

            /* Botones en azul oscuro y letras en blanco */
            button[kind="primary"],
            button[kind="secondary"],
            div.stButton > button,
            div.stDownloadButton > button,
            div[data-testid="stFormSubmitButton"] > button {
                background-color: #0A365C !important;
                background-image: none !important;
                color: #FFFFFF !important;
                border: 1px solid #07253F !important;
                border-radius: 8px !important;
                font-family: 'Montserrat', sans-serif !important;
                font-size: 0.92rem !important;
                font-weight: 700 !important;
                padding: 0.55rem 1.2rem !important;
                box-shadow: 0 2px 4px rgba(10, 54, 92, 0.2) !important;
                transition: all 0.2s ease !important;
            }

            button[kind="primary"] *,
            button[kind="secondary"] *,
            div.stButton > button *,
            div.stDownloadButton > button *,
            div[data-testid="stFormSubmitButton"] > button * {
                color: #FFFFFF !important;
                -webkit-text-fill-color: #FFFFFF !important;
                font-weight: 700 !important;
            }

            button[kind="primary"]:hover,
            button[kind="secondary"]:hover,
            div.stButton > button:hover,
            div.stDownloadButton > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                background-color: #0E487A !important;
                border-color: #0A365C !important;
                box-shadow: 0 4px 8px rgba(10, 54, 92, 0.3) !important;
            }

            /* Slider */
            div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
                background-color: #7DD3FC !important;
                border: 2px solid #38BDF8 !important;
                box-shadow: 0 1px 3px rgba(14, 165, 233, 0.25) !important;
                width: 14px !important;
                height: 14px !important;
            }
            div[data-testid="stSlider"] [data-baseweb="slider"] div > div {
                background-color: #BAF2FD !important;
            }
            div[data-testid="stSlider"] [data-testid="stTickBar"] span,
            div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
                font-size: 0.78rem !important;
                color: #64748B !important;
            }
            div[data-testid="stSlider"] div:has(> div[role="slider"]) ~ div {
                color: #0284C7 !important;
                font-size: 0.85rem !important;
                font-weight: 600 !important;
            }

            /* Tags del Multiselect */
            span[data-baseweb="tag"] {
                background-color: #D9F9FF !important;
                border: 1px solid #7DD3FC !important;
                border-radius: 6px !important;
                height: auto !important;
                padding: 3px 7px !important;
            }
            span[data-baseweb="tag"] span {
                color: #0F172A !important;
                font-family: 'Montserrat', sans-serif !important;
                font-size: 0.8rem !important;
                font-weight: 500 !important;
            }
            span[data-baseweb="tag"] svg {
                fill: #0F172A !important;
            }
            span[data-baseweb="tag"]:hover {
                background-color: #BAF2FD !important;
            }

            /* Pestañas (Tabs) */
            div[data-testid="stTabs"] {
                overflow: visible !important;
                margin-top: 1rem !important;
                margin-bottom: 1.5rem !important;
            }

            div[data-testid="stTabs"] > div[role="tablist"],
            .stTabs > div[role="tablist"] {
                width: 100vw !important;
                position: relative !important;
                left: 50% !important;
                right: 50% !important;
                margin-left: -50vw !important;
                margin-right: -50vw !important;
                background-color: #E0F2FE !important;
                border-top: 1px solid #BAE6FD !important;
                border-bottom: 1px solid #BAE6FD !important;
                border-left: none !important;
                border-right: none !important;
                border-radius: 0px !important;
                padding: 0 2rem !important;
                display: flex !important;
                gap: 0px !important;
                box-sizing: border-box !important;
            }

            div[data-testid="stTabs"] button[role="tab"],
            div[data-testid="stTabsTab"],
            .stTabs button {
                flex: 1 1 0 !important;
                background-color: transparent !important;
                border: none !important;
                border-radius: 0px !important;
                padding: 12px 18px !important;
                font-family: 'Montserrat', sans-serif !important;
                font-size: 0.9rem !important;
                text-align: center !important;
                box-shadow: none !important;
                transition: all 0.2s ease !important;
            }

            div[data-testid="stTabs"] button[role="tab"] p,
            div[data-testid="stTabsTab"] p,
            .stTabs button p,
            div[data-testid="stTabs"] button[role="tab"] span {
                color: #0F172A !important;
                font-weight: 600 !important;
                margin: 0 !important;
            }

            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
            div[data-testid="stTabsTab"][aria-selected="true"],
            .stTabs button[aria-selected="true"] {
                background-color: #BAE6FD !important;
                border-bottom: 3px solid #0369A1 !important;
            }

            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] div,
            div[data-testid="stTabsTab"][aria-selected="true"] p,
            div[data-testid="stTabsTab"][aria-selected="true"] span,
            .stTabs button[aria-selected="true"] p,
            .stTabs button[aria-selected="true"] span {
                color: #0A365C !important;
                font-weight: 800 !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
            div[data-testid="stTabs"] [data-baseweb="tab-border"] {
                display: none !important;
            }

            div[data-stale="true"],
            .stApp [data-stale="true"] {
                opacity: 1 !important;
                filter: none !important;
                transition: none !important;
            }

            [data-stale="true"] {
                visibility: hidden !important;
                height: 0 !important;
                overflow: hidden !important;
            }

            section[data-testid="stSidebar"] [data-testid="stProgress"] > div {
                background-color: #CBD5E1 !important;
                border-radius: 9999px !important;
                height: 8px !important;
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()

# Encabezado principal
st.title("SIMD Lab Gender Auditor")
st.markdown("Plataforma para el cálculo de la presencia y protagonismo de la mujer en una entidad o empresa en base a varias fuentes de información.")

# 1. Lectura del entorno configurado en la libreta (por defecto 'test_ui')
data_mode = os.getenv("AUDIT_DATA_MODE", "test_ui").strip().lower()
test_ui = (data_mode == "test_ui")

# Indicador visual informativo en la barra lateral
with st.sidebar:
    st.markdown("---")
    st.markdown("##### Entorno de Ejecución (Notebook)")
    badge_bg = "#DCFCE7" if test_ui else "#E0F2FE"
    badge_color = "#15803D" if test_ui else "#0369A1"
    mode_label = "Modo Simulación (test_ui)" if test_ui else "Modelos Reales"
    st.markdown(
        f"""
        <div style="background-color: {badge_bg}; color: {badge_color}; border-radius: 6px; padding: 6px 12px; font-weight: 700; font-size: 0.82rem; text-align: center;">
            {mode_label}
        </div>
        """,
        unsafe_allow_html=True
    )

# Parámetros devueltos por el sidebar (empresa, top_n, fuentes y botón de ejecución)
company, top_n, selected_sources, run_btn = render_sidebar()

# 2. Inicialización de Session State para persistencia entre interacciones
if "audit_executed" not in st.session_state:
    st.session_state.audit_executed = False
if "docs" not in st.session_state:
    st.session_state.docs = None
if "score_data" not in st.session_state:
    st.session_state.score_data = None
if "active_company" not in st.session_state:
    st.session_state.active_company = company


@st.cache_resource
def load_scorer():
    return CorporateGenderScorer()

scorer = load_scorer()


@st.cache_data(show_spinner=False)
def get_cached_documents(_test_ui: bool, company_name: str, n_docs: int, sources_tuple: tuple):
    if _test_ui:
        from tests.test_xai_fairness_mock import generate_mock_dataset
        return generate_mock_dataset(company=company_name)
    else:
        try:
            return fetch_analyzed_documents(company_name, n_docs, list(sources_tuple))
        except Exception as exc:
            st.error(f"Fallo en el pipeline real: {type(exc).__name__}: {exc}")
            st.stop()

@st.cache_data(show_spinner=False)
def compute_cached_scores(_scorer, docs_payload):
    return _scorer.predict_importance_score(docs_payload)


# 3. Disparo exclusivo tras pulsar el botón
if run_btn:
    with st.spinner("Ejecutando pipeline y calculando puntuaciones multimodales..."):
        docs_computed = get_cached_documents(test_ui, company, top_n, tuple(selected_sources))
        score_data_computed = compute_cached_scores(scorer, docs_computed)

        # Almacenar en session_state para mantener la reactividad al navegar entre tabs
        st.session_state.docs = docs_computed
        st.session_state.score_data = score_data_computed
        st.session_state.active_company = company
        st.session_state.perfil_lexico = get_lexical_profile(company, top_n, selected_sources, docs_computed)
        st.session_state.audit_executed = True

# 4. Renderizado condicional
if st.session_state.audit_executed:
    tab1, tab2, tab3 = st.tabs([
        "1. Ponderación de Métricas Globales",
        "2. Explicabilidad (SHAP & LIME)",
        "3. Gobernanza y Cumplimiento EU AI Act"
    ])

    with tab1:
        render_esg_dashboard(st.session_state.active_company, st.session_state.score_data,
                             st.session_state.get("perfil_lexico"))
    with tab2:
        render_xai_tab(st.session_state.docs, st.session_state.score_data, scorer)
    with tab3:
        render_ai_act_tab(st.session_state.active_company, st.session_state.score_data)
else:
    # Mensaje de bienvenida inicial (antes de pulsar el botón)
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1.5px dashed #CBD5E1; border-radius: 12px; padding: 48px 24px; text-align: center; margin-top: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <h3 style="color: #0A365C; font-family: 'Merriweather', serif; margin-bottom: 8px;">Auditoría no iniciada</h3>
            <p style="color: #64748B; font-family: 'Montserrat', sans-serif; font-size: 0.92rem; max-width: 620px; margin: 0 auto;">
                Configura los parámetros de búsqueda en la barra lateral izquierda y pulsa el botón para comenzar el procesamiento multimodal.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )