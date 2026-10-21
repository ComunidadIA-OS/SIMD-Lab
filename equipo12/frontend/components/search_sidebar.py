import psutil
import subprocess
import streamlit as st
from src.common.schemas import MediaSource
def _render_bar(label: str, value_text: str, percentage: float):
    """Dibuja una etiqueta y una única barra de progreso proporcional."""
    pct_clamped = max(0.0, min(percentage, 100.0))
    st.sidebar.markdown(
        f"""
          <div style="margin-bottom: 10px; font-family: 'Montserrat', sans-serif;">
              <div style="font-size: 0.8rem; color: #1E293B; margin-bottom: 4px;">
                  <strong>{label}:</strong> <span style="color: #64748B;">{value_text}</span>
              </div>
              <div style="
                  background-color: #E2E8F0;
                  border-radius: 999px;
                  height: 7px;
                  width: 100%;
                  overflow: hidden;
              ">
                  <div style="
                      background-color: #0A365C;
                      width: {pct_clamped}%;
                      height: 100%;
                      border-radius: 999px;
                  "></div>
              </div>
          </div>
          """,
        unsafe_allow_html=True,
    )
def render_resource_monitor():
    st.sidebar.divider()
    st.sidebar.markdown("#### Monitorización del Sistema")

    # 1. CPU
    cpu_pct = psutil.cpu_percent()
    _render_bar("Uso CPU", f"{cpu_pct:.1f}%", cpu_pct)

    # 2. RAM
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    _render_bar(
        "Uso RAM",
        f"{ram.percent:.1f}% ({ram_used_gb:.1f} / {ram_total_gb:.1f} GB)",
        ram.percent,
    )

    # 3. GPU (Carga)
    gpu_pct = 0.0  # Conectar a la variable real de GPU
    _render_bar("Uso GPU", f"{gpu_pct:.1f}%", gpu_pct)

    # 4. VRAM
    vram_used_mb = 0.0
    vram_total_mb = 16000.0
    vram_pct = (
        (vram_used_mb / vram_total_mb) * 100.0 if vram_total_mb > 0 else 0.0
    )
    _render_bar(
        "VRAM",
        f"{vram_pct:.1f}% ({vram_used_mb:.0f} / {vram_total_mb:.0f} MB)",
        vram_pct,
    )

    # 5. Potencia
    power_w = 29.1
    max_power_w = 250.0
    power_pct = (power_w / max_power_w) * 100.0
    _render_bar("Consumo Potencia", f"{power_w:.1f} W", power_pct)

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
    
    render_resource_monitor()
    
    return company, top_n, selected_sources, run_btn