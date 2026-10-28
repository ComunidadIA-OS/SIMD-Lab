import html
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

from src.explainability.lime_explainer import LimeMultimodalEngine
from src.explainability.shap_explainer import ShapExplanationEngine
from src.explainability.text_evidence import explain_text


def _fmt(v, pct: bool = False) -> str:
    """Formatea una métrica; "sin dato" si el analizador no la ha podido calcular (None)."""
    if v is None:
        return "sin dato"
    return f"{v * 100:.1f}%" if pct else f"{v:.2f}"


@st.cache_resource(show_spinner="Cargando el analizador de texto…")
def _analizador_texto():
    """El mismo analizador de texto que usa la auditoría (spaCy), cargado una sola vez."""
    from src.api.main_controller import _get_pipeline
    return _get_pipeline().texto


@st.cache_data(show_spinner="Buscando menciones a personas…", max_entries=64)
def _menciones_html(texto: str, empresa: str) -> str:
    return explain_text(texto, _analizador_texto(), company_query=empresa)


def _extract_doc_texts(doc) -> list:
    """Extrae todos los fragmentos o textos disponibles (objetos o strings)."""
    texts = []
    samples = getattr(doc, "text_samples", None)
    if isinstance(samples, (list, tuple)):
        for s in samples:
            text_val = getattr(s, "text", str(s)).strip()
            if text_val:
                texts.append(s)  # Guardamos el objeto o texto completo

    snippet = getattr(doc, "raw_text_snippet", None)
    if snippet and str(snippet).strip():
        # Evitar duplicados si el snippet ya está
        if not any(getattr(s, "text", str(s)) == str(snippet).strip() for s in texts):
            texts.insert(0, snippet)
    return texts


def _extract_doc_audio_samples(doc) -> list:
    """Extrae referencias o objetos de audio analizados."""
    audios = []
    samples = getattr(doc, "media_audio_samples", None)
    if isinstance(samples, (list, tuple)):
        for s in samples:
            if s:
                audios.append(s)
    return audios


def _extract_doc_video_samples(doc) -> list:
    """Extrae referencias o objetos de vídeo analizados."""
    videos = []
    samples = getattr(doc, "media_video_samples", None)
    if isinstance(samples, (list, tuple)):
        for s in samples:
            if s:
                videos.append(s)
    return videos


def _extract_doc_images(doc) -> list:
    """Extrae de forma segura las referencias o objetos de imagen asociados."""
    found_images = []

    samples = getattr(doc, "media_image_samples", None)
    if isinstance(samples, (list, tuple)):
        for item in samples:
            path_val = getattr(item, "image_path", str(item))
            if path_val and Path(str(path_val)).exists():
                found_images.append(item)

    for attr in ["images", "image_paths", "gallery", "photos"]:
        val = getattr(doc, attr, None)
        if isinstance(val, (list, tuple)):
            for item in val:
                path_val = getattr(item, "image_path", str(item))
                if path_val and Path(str(path_val)).exists() and item not in found_images:
                    found_images.append(item)

    for attr in ["image_path", "image", "thumbnail"]:
        val = getattr(doc, attr, None)
        path_val = getattr(val, "image_path", str(val)) if val else None
        if path_val and Path(str(path_val)).exists() and val not in found_images:
            found_images.append(val)

    return found_images


def _detect_doc_modalities(doc) -> list[str]:
    """Determina con rigor las modalidades activas respetando el campo available_modalities."""
    if hasattr(doc, "available_modalities") and doc.available_modalities:
        return list(doc.available_modalities)

    modalities = set()
    if _extract_doc_texts(doc):
        modalities.add("Texto")
    if _extract_doc_images(doc):
        modalities.add("Imagen")
    if getattr(doc, "video_metrics", None) or _extract_doc_video_samples(doc):
        modalities.add("Vídeo")
    if getattr(doc, "audio_metrics", None) or _extract_doc_audio_samples(doc):
        modalities.add("Audio")

    order = ["Texto", "Vídeo", "Audio", "Imagen"]
    return [m for m in order if m in modalities]


def render_xai_tab(docs, score_data, scorer):
    st.subheader("Explicabilidad (SHAP & LIME)")

    if not docs:
        st.info("No hay documentos disponibles en el dataset auditado.")
        return

    shap_engine = ShapExplanationEngine(scorer)
    lime_engine = LimeMultimodalEngine()
    features_df = score_data.get("features_only")

    col_main_view, col_right_panel = st.columns([3.5, 1.2], gap="large")

    with col_right_panel:
        with st.container(border=True):
            st.markdown("#### Control de Explicabilidad")
            scope = st.radio(
                "Nivel de inspección:",
                options=["1. Impacto Global (Dataset)", "2. Desglose Local por Documento"],
                label_visibility="collapsed",
                key="xai_scope_selector",
            )
            st.divider()

            selected_idx = 0
            selected_txt_idx = 0
            selected_img_idx = 0
            selected_aud_idx = 0
            selected_vid_idx = 0
            doc_modalities = []
            doc_texts = []
            doc_images = []
            doc_audios = []
            doc_videos = []

            if scope == "2. Desglose Local por Documento":
                def format_doc_title(i: int) -> str:
                    d = docs[i]
                    source_str = getattr(d.source, "value", str(d.source))
                    title_str = getattr(d, "title", f"Documento_{i}")
                    return f"{source_str}: {title_str[:40]}..."

                selected_idx = st.selectbox(
                    "Documento auditado:",
                    options=list(range(len(docs))),
                    format_func=format_doc_title,
                    key="xai_selected_doc_idx",
                )

                curr_doc = docs[selected_idx]

                doc_modalities = _detect_doc_modalities(curr_doc)
                doc_texts = _extract_doc_texts(curr_doc)
                doc_images = _extract_doc_images(curr_doc)
                doc_audios = _extract_doc_audio_samples(curr_doc)
                doc_videos = _extract_doc_video_samples(curr_doc)

                st.markdown("##### Modalidades activas detectadas")
                if doc_modalities:
                    badges = " ".join([
                        f"<span style='background:#E0F2FE;color:#0A365C;padding:3px 8px;border-radius:6px;font-size:0.75rem;font-weight:700;'>{m}</span>"
                        for m in doc_modalities
                    ])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.caption("Sin modalidades detectadas en este documento.")

                st.write("")
                selected_modality = st.selectbox(
                    "Filtrar por formato:",
                    options=doc_modalities if doc_modalities else ["Texto"],
                    key="xai_selected_modality",
                )

                # Métodos interpretativos disponibles
                method_options = ["SHAP Waterfall (Atribución Local)"]
                if selected_modality == "Texto" and len(doc_texts) > 0:
                    method_options.append("Menciones detectadas en el texto")
                elif selected_modality == "Imagen" and len(doc_images) > 0:
                    method_options.append("LIME Imagen (Superpíxeles)")

                selected_method = st.radio(
                    "Método XAI:",
                    options=method_options,
                    key="xai_local_method",
                )

                # Selectores de sub-muestra (fragmento de texto o fotografía)
                if selected_method.startswith(("LIME", "Menciones")):
                    st.divider()
                    if selected_modality == "Texto" and len(doc_texts) > 0:
                        selected_txt_idx = st.selectbox(
                            "Fragmento de texto:",
                            options=list(range(len(doc_texts))),
                            format_func=lambda i: f"Texto {i+1}: {getattr(doc_texts[i], 'text', str(doc_texts[i]))[:30]}...",
                            key="xai_lime_txt_subidx",
                        )
                    elif selected_modality == "Imagen" and len(doc_images) > 0:
                        selected_img_idx = st.selectbox(
                            "Fotografía específica para LIME:",
                            options=list(range(len(doc_images))),
                            format_func=lambda i: f"Fotografía {i+1} de {len(doc_images)}",
                            key="xai_lime_img_subidx",
                        )

                # Selectores para pistas multimedia generales si procede
                if selected_modality == "Audio" and len(doc_audios) > 0:
                    selected_aud_idx = st.selectbox(
                        "Pista de audio a auditar:",
                        options=list(range(len(doc_audios))),
                        format_func=lambda i: f"Pista {i+1}: {Path(getattr(doc_audios[i], 'audio_path', str(doc_audios[i]))).name}",
                        key="xai_selected_aud_subidx",
                    )

                if selected_modality == "Vídeo" and len(doc_videos) > 0:
                    selected_vid_idx = st.selectbox(
                        "Clip de vídeo a auditar:",
                        options=list(range(len(doc_videos))),
                        format_func=lambda i: f"Clip {i+1}: {Path(getattr(doc_videos[i], 'video_path', str(doc_videos[i]))).name}",
                        key="xai_selected_vid_subidx",
                    )

    with col_main_view:
        with st.container(border=True):
            if scope == "1. Impacto Global (Dataset)":
                st.markdown("#### Impacto Global de Variables en el Score (SHAP Bar)")
                st.caption("Ponderación media absoluta del impacto de cada variable sobre las decisiones del modelo:")
                if features_df is not None and not features_df.empty:
                    fig_global = shap_engine.plot_global_summary(features_df)
                    st.pyplot(fig_global)
                else:
                    st.warning("No se dispone de la matriz de variables para el cálculo global.")
            else:
                curr_doc = docs[selected_idx]
                doc_id = getattr(curr_doc, "doc_id", f"doc_{selected_idx}")
                src_val = getattr(curr_doc.source, "value", str(curr_doc.source))

                st.markdown(f"#### Auditoría Local: {selected_modality} — `{doc_id}` ({src_val})")

                if selected_method.startswith("SHAP Waterfall"):
                    st.markdown(
                        """
                        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-bottom: 1.2rem; font-family: 'Montserrat', sans-serif; font-size: 0.82rem; color: #1E293B; line-height: 1.6;">
                            <strong style="font-size: 0.88rem; color: #0F172A;">Guía de lectura de la auditoría SHAP (Atribución Local):</strong>
                            <ul style="margin: 6px 0 8px 0; padding-left: 20px;">
                                <li><strong style="color: #64748B;">Valor gris a la izquierda (ej. 0.85 =)</strong>: Magnitud real observada de la métrica en este documento concreto.</li>
                                <li><strong style="color: #F43F5E;">■ Barras rojas (+)</strong>: Variables que incrementan la relevancia de liderazgo femenino por encima de la media base.</li>
                                <li><strong style="color: #0EA5E9;">■ Barras azules (−)</strong>: Factores que restan puntuación o penalizan la visibilidad del perfil.</li>
                                <li><strong>f(x)</strong>: Puntuación o score final asignado al documento tras aplicar todas las desviaciones.</li>
                                <li><strong>E[f(x)]</strong>: Valor base esperado (puntuación media de referencia en el dataset de entrenamiento).</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    prefix_map = {"Texto": "txt_", "Vídeo": "vid_", "Audio": "aud_", "Imagen": "img_"}
                    fig_local = shap_engine.plot_local_waterfall(
                        features_df,
                        doc_index=selected_idx,
                        modality_prefix=prefix_map.get(selected_modality)
                    )
                    st.pyplot(fig_local)

                    if selected_modality == "Audio":
                        st.divider()
                        st.markdown("##### Verificación de Muestra Acústica (Supervisión Humana)")
                        
                        # Extraer la muestra y sus métricas individuales o globales
                        target_aud_sample = doc_audios[selected_aud_idx] if doc_audios else None
                        audio_path_val = getattr(target_aud_sample, "audio_path", str(target_aud_sample)) if target_aud_sample else ""
                        sample_metrics = getattr(target_aud_sample, "metrics", getattr(curr_doc, "audio_metrics", None))

                        col_aud_play, col_aud_metrics = st.columns([1.5, 1], gap="medium")
                        with col_aud_play:
                            if audio_path_val:
                                st.caption(f"Pista auditada: `{Path(audio_path_val).name}`")
                                if Path(audio_path_val).exists() and Path(audio_path_val).stat().st_size > 0:
                                    st.audio(audio_path_val)
                                else:
                                    st.info(f"Referencia registrada en metadatos: `{audio_path_val}`")
                            else:
                                st.caption("No existen ficheros de audio locales vinculados.")
                        with col_aud_metrics:
                            st.markdown("**Métricas Acústicas Extraídas:**")
                            if sample_metrics:
                                st.write(f"- **Ratio Tiempo Hablado:** `{_fmt(getattr(sample_metrics, 'female_speaking_time_ratio', None), pct=True)}`")
                                st.write(f"- **Turnos de Habla:** `{getattr(sample_metrics, 'female_turn_count', 0)}`")
                                st.write(f"- **Apertura Femenina:** `{getattr(sample_metrics, 'female_opened_score', 0.0):.2f}`")
                                st.write(f"- **Ausencia de Interrupciones:** `{_fmt(getattr(sample_metrics, 'no_interruption_score', None))}`")

                    elif selected_modality == "Vídeo":
                        st.divider()
                        st.markdown("##### Verificación de Muestra de Vídeo (Supervisión Humana)")
                        
                        target_vid_sample = doc_videos[selected_vid_idx] if doc_videos else None
                        video_path_val = getattr(target_vid_sample, "video_path", str(target_vid_sample)) if target_vid_sample else ""
                        # Los vídeos suelen depender de audio_track / frame_set_visuals o métricas globales de vídeo
                        target_vid_sample = doc_videos[selected_vid_idx] if doc_videos else None
                        video_path_val = getattr(target_vid_sample, "video_path", str(target_vid_sample)) if target_vid_sample else ""
                        vm = getattr(target_vid_sample, "metrics", getattr(curr_doc, "video_metrics", None))

                        col_vid_play, col_vid_metrics = st.columns([1.5, 1], gap="medium")
                        with col_vid_play:
                            if video_path_val:
                                st.caption(f"Clip auditado: `{Path(video_path_val).name}`")
                                if Path(video_path_val).exists() and Path(video_path_val).stat().st_size > 0:
                                    st.video(video_path_val)
                                elif str(video_path_val).startswith("http"):
                                    st.video(video_path_val)
                                else:
                                    st.info(f"Evidencia registrada en metadatos de la fuente:\n\n`{video_path_val}`")
                            else:
                                st.caption("No existen clips de vídeo locales vinculados.")
                        with col_vid_metrics:
                            st.markdown("**Tensores Cinematográficos Extraídos:**")
                            if vm:
                                st.write(f"- **Presencia en Pantalla:** `{getattr(vm, 'female_screen_time_ratio', 0.0)*100:.1f}%`")
                                st.write(f"- **Flujo Óptico (Movimiento):** `{_fmt(getattr(vm, 'optical_flow_motion_score', None))}`")
                                st.write(f"- **Dominancia Cinética:** `{_fmt(getattr(vm, 'kinetic_dominance_ratio', None))}`")
                                st.write(f"- **Sincronía Labio-Voz:** `{_fmt(getattr(vm, 'active_speaker_sync_score', None))}`")

                elif selected_method.startswith("Menciones"):
                    target_txt_sample = doc_texts[selected_txt_idx] if doc_texts else ""
                    target_text = getattr(target_txt_sample, "text", str(target_txt_sample))
                    if target_text:
                        st.caption(
                            "Personas que el analizador de texto ha encontrado en este fragmento (nombres, cargos y "
                            "pronombres), con su género y si son sujeto de la acción o están citadas. Las métricas "
                            "de texto se calculan solo con estas menciones."
                        )
                        try:
                            st.html(_menciones_html(target_text, getattr(curr_doc, "company_query", "") or ""))
                        except Exception as e:
                            st.error(f"No se ha podido analizar el fragmento: {e}")
                    else:
                        st.info("El documento no contiene fragmentos textuales para auditar.")

                elif selected_method.startswith("LIME Imagen"):
                    if doc_images:
                        target_img_sample = doc_images[selected_img_idx]
                        target_image = getattr(target_img_sample, "image_path", str(target_img_sample))
                        st.caption("Superpíxeles que determinaron el encuadre y centralidad visual (contornos amarillos):")
                        try:
                            explained_img = lime_engine.explain_image_instance(target_image)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Muestra Efímera Original**")
                                st.image(target_image, use_container_width=True)
                            with c2:
                                st.markdown("**Regiones Decisivas (LIME Superpíxeles)**")
                                st.image(
                                    explained_img,
                                    caption="Regiones que condicionaron la predicción de encuadre",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Error procesando la muestra visual: {e}")
                    else:
                        st.info("No existen muestras de imagen disponibles para auditar con LIME.")