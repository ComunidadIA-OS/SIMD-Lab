import psutil
import subprocess
import streamlit as st
from src.common.schemas import MediaSource

def render_sidebar():
    st.sidebar.title("Parámetros de Búsqueda")
    st.sidebar.markdown("Introduce la entidad o corporación cuyas fuentes deseas auditar.")
    
    company = st.sidebar.text_input("Empresa o Palabra Clave", value="Iberdrola")
    top_n = st.sidebar.slider("Top N documentos relevantes", min_value=1, max_value=20, value=5)
    
    all_sources = [s.value for s in MediaSource]
    selected_sources = st.sidebar.multiselect(
        "Fuentes de datos activas", options=all_sources, default=all_sources
    )
    run_btn = st.sidebar.button("Ejecutar Auditoría", type="primary", width="stretch")
    
    return company, top_n, selected_sources, run_btn