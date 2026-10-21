import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _style_plotly(fig):
    """Aplica fondo blanco, tipografía formal Merriweather y rejilla gris suave."""
    fig.update_layout(
        height=450,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Merriweather, serif", color="#1E293B", size=13),
        title_font=dict(family="Merriweather, serif", color="#0F172A", size=16),
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
    )
    return fig


def _safe_mean(df: pd.DataFrame, possible_cols: list[str]) -> float:
    """Extrae el promedio normalizado (0-100) comprobando los posibles nombres de columna."""
    for col in possible_cols:
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                return round(float(s.mean()) * 100.0, 1)
    return 0.0


@st.fragment
def _render_interactive_charts(
    company: str,
    score_data: dict,
    df: pd.DataFrame,
    df_calc: pd.DataFrame,
    weights: dict,
):
    """Fragmento interactivo para proyectar análisis por plataforma y modalidades."""
    mod = score_data.get("by_modality", {})
    col_main_chart, col_right_panel = st.columns([3.5, 1.2], gap="large")

    with col_right_panel:
        with st.container(border=True):
            st.markdown("#### Desglose Multimodal")
            selected_chart = st.radio(
                "Modalidad activa:",
                options=[
                    "1. Distribución de Contenido por Medio",
                    "2. Métricas de Vídeo y Movimiento",
                    "3. Métricas de Imagen y Fotografía",
                    "4. Métricas de Texto y Léxico",
                    "5. Métricas de Audio y Turnos",
                ],
                label_visibility="collapsed",
                key="esg_dashboard_selected_chart",
            )
            st.divider()
            st.caption(
                "Las métricas se normalizan al rango porcentual (0–100%) para"
                " posibilitar auditorías cruzadas sin sesgo de escala."
            )

    with col_main_chart:
        with st.container(border=True):
            # -------------------------------------------------------------
            # 1. MATRIZ DE DISTRIBUCIÓN POR MEDIO Y MODALIDAD
            # -------------------------------------------------------------
            if selected_chart == "1. Distribución de Contenido por Medio":
                total_act = max(
                    df_calc["act_Texto"].sum()
                    + df_calc["act_Audio"].sum()
                    + df_calc["act_Imagen"].sum()
                    + df_calc["act_Vídeo"].sum(),
                    1.0,
                )

                w_texto = round((df_calc["act_Texto"].sum() / total_act) * 100.0, 1)
                w_audio = round((df_calc["act_Audio"].sum() / total_act) * 100.0, 1)
                w_imagen = round((df_calc["act_Imagen"].sum() / total_act) * 100.0, 1)
                w_video = round((df_calc["act_Vídeo"].sum() / total_act) * 100.0, 1)

                columnas_matriz = ["Docs", "Texto", "Audio", "Imagen", "Vídeo"]
                lista_plataformas = list(
                    df["source"].unique()
                    if "source" in df.columns
                    else score_data.get("by_source", {}).keys()
                )
                lista_completa = lista_plataformas + ["Total"]

                max_src_val = 1.0
                for src in lista_plataformas:
                    src_sub = (
                        df_calc[df_calc["source"] == src]
                        if "source" in df.columns
                        else df_calc
                    )
                    for col_act in ["act_Texto", "act_Audio", "act_Imagen", "act_Vídeo"]:
                        v = (src_sub[col_act].sum() / total_act) * 100.0
                        if v > max_src_val:
                            max_src_val = v

                matriz_z, matriz_textos, annotations = [], [], []

                for row_name in lista_completa:
                    es_total = row_name == "Total"
                    fila_z, fila_txt = [], []

                    if not es_total:
                        src_sub = (
                            df_calc[df_calc["source"] == row_name]
                            if "source" in df.columns
                            else df_calc
                        )
                        cnt_docs = len(src_sub)
                        fila_z.append(0.0)
                        fila_txt.append(f"{cnt_docs} doc{'s' if cnt_docs != 1 else ''}")

                        for col_act in ["act_Texto", "act_Audio", "act_Imagen", "act_Vídeo"]:
                            act_val = round((src_sub[col_act].sum() / total_act) * 100.0, 1)
                            z_norm = (
                                2.0 + (act_val / max_src_val) * 46.0
                                if act_val > 0
                                else 0.0
                            )
                            fila_z.append(z_norm)
                            fila_txt.append(f"{act_val:.1f}%")
                    else:
                        fila_z.append(0.0)
                        fila_txt.append(f"{len(df)} docs")
                        max_w = max(w_texto, w_audio, w_imagen, w_video, 1.0)
                        for w_val in [w_texto, w_audio, w_imagen, w_video]:
                            z_norm = 60.0 + (w_val / max_w) * 40.0
                            fila_z.append(z_norm)
                            fila_txt.append(f"{w_val:.1f}%")

                    matriz_z.append(fila_z)
                    matriz_textos.append(fila_txt)

                    for x_idx, col_name in enumerate(columnas_matriz):
                        txt = fila_txt[x_idx]
                        if col_name == "Docs":
                            f_color = "#0F172A"
                        elif es_total:
                            f_color = "#FFFFFF"
                        else:
                            val_pct = (
                                float(txt.replace("%", ""))
                                if "%" in txt
                                else 0.0
                            )
                            f_color = (
                                "#FFFFFF"
                                if (val_pct / max_src_val) >= 0.50
                                else "#0F172A"
                            )

                        annotations.append(
                            dict(
                                x=col_name,
                                y=row_name,
                                text=f"<b>{txt}</b>",
                                showarrow=False,
                                font=dict(
                                    family="Montserrat",
                                    size=13 if es_total or col_name != "Docs" else 12,
                                    color=f_color,
                                ),
                            )
                        )

                colorscale = [
                    [0.00, "#F1F5F9"],
                    [0.02, "#F8FAFC"],
                    [0.15, "#BAE6FD"],
                    [0.30, "#38BDF8"],
                    [0.48, "#0A365C"],
                    [0.49, "#E2E8F0"],
                    [0.59, "#E2E8F0"],
                    [0.60, "#64748B"],
                    [0.80, "#475569"],
                    [1.00, "#1E293B"],
                ]

                fig_matrix = go.Figure(
                    data=go.Heatmap(
                        z=matriz_z,
                        x=columnas_matriz,
                        y=lista_completa,
                        zmin=0.0,
                        zmax=100.0,
                        colorscale=colorscale,
                        showscale=False,
                        xgap=5,
                        ygap=5,
                        hoverinfo="none",
                    )
                )

                fig_matrix.update_layout(
                    height=460,
                    margin=dict(t=65, b=30, l=10, r=20),
                    annotations=annotations,
                    title=dict(
                        text=f"Aporte por Medio y Modalidad (% Dataset) — {company}",
                        font=dict(family="Merriweather, serif", size=16, color="#0F172A"),
                        x=0.0,
                        y=0.97,
                    ),
                    xaxis=dict(
                        title="",
                        side="top",
                        tickfont=dict(
                            family="Montserrat",
                            size=13,
                            color="#0F172A",
                        ),
                    ),
                    yaxis=dict(
                        title="",
                        autorange="reversed",
                        tickfont=dict(
                            family="Montserrat", size=12, color="#1E293B"
                        ),
                    ),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                )
                st.plotly_chart(
                    fig_matrix,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # -------------------------------------------------------------
            # 2. MÉTRICAS DE VÍDEO
            # -------------------------------------------------------------
            elif selected_chart == "2. Métricas de Vídeo y Movimiento":
                v_opt = _safe_mean(df, ["vid_optical_flow_motion_score", "optical_flow_motion_score"])
                v_kin = _safe_mean(df, ["vid_kinetic_dominance_ratio", "kinetic_dominance_ratio"])
                v_sil = _safe_mean(df, ["vid_silhouette_change_histogram_score", "silhouette_change_histogram_score"])
                v_scr = _safe_mean(df, ["vid_female_screen_time_ratio", "female_screen_time_ratio"])
                v_spk = _safe_mean(df, ["vid_active_speaker_sync_score", "active_speaker_sync_score"])

                vid_data = pd.DataFrame({
                    "Métrica": [
                        "Flujo Óptico (Movimiento) [w=0.15]",
                        "Dominancia Cinética [w=0.15]",
                        "Histograma de Silueta [w=0.15]",
                        "Tiempo en Pantalla [w=0.35]",
                        "Sincronía Labio-Voz [w=0.20]",
                    ],
                    "Valor (%)": [v_opt, v_kin, v_sil, v_scr, v_spk],
                    "TextoBarra": [f"{v_opt}%", f"{v_kin}%", f"{v_sil}%", f"{v_scr}%", f"{v_spk}%"],
                })
                fig = px.bar(
                    vid_data,
                    x="Valor (%)",
                    y="Métrica",
                    orientation="h",
                    text="TextoBarra",
                    range_x=[0, 110],
                    color_discrete_sequence=["#38BDF8"],
                    title="Análisis de Vídeo: Dinámica de Movimiento, Siluetas y Pantalla",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(_style_plotly(fig), use_container_width=True)

                sum_vid = mod.get(
                    "Vídeo",
                    round(0.15 * v_opt + 0.15 * v_kin + 0.15 * v_sil + 0.35 * v_scr + 0.20 * v_spk, 1)
                )

                st.markdown("---")
                st.markdown("##### Fórmula para el cálculo de la puntuación en vídeos")
                st.latex(r"\text{Score}_{\text{Vídeo}} = \sum_{i=1}^{n} w_i \cdot \text{Métrica}_i \quad \text{donde} \quad \sum w_i = 1.0")
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px 16px; font-family: 'Montserrat', sans-serif; font-size: 0.85rem; color: #1E293B; margin-bottom: 1.5rem;">
                        <strong>Suma de componentes:</strong><br>
                        <code>(0.35 × {v_scr}) + (0.20 × {v_spk}) + (0.15 × {v_kin}) + (0.15 × {v_sil}) + (0.15 × {v_opt})</code><br>
                        = <code>{round(0.35*v_scr, 2)} + {round(0.20*v_spk, 2)} + {round(0.15*v_kin, 2)} + {round(0.15*v_sil, 2)} + {round(0.15*v_opt, 2)}</code>
                        = <strong style="color: #0A365C; font-size: 0.95rem;">{sum_vid} / 100</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------------------
            # 3. MÉTRICAS DE IMAGEN
            # -------------------------------------------------------------
            elif selected_chart == "3. Métricas de Imagen y Fotografía":
                i_gen = _safe_mean(df, ["img_visual_gender_female_ratio", "visual_gender_female_ratio"])
                i_scl = _safe_mean(df, ["img_shot_scale_importance_score", "shot_scale_importance_score"])
                i_ang = _safe_mean(df, ["img_camera_angle_power_score", "camera_angle_power_score"])
                i_thd = _safe_mean(df, ["img_rule_of_thirds_alignment", "rule_of_thirds_alignment"])
                i_shd = _safe_mean(df, ["img_shadow_lighting_prominence", "shadow_lighting_prominence"])
                i_per = _safe_mean(df, ["img_cross_image_persistence_ratio", "cross_image_persistence_ratio"])
                i_cen = _safe_mean(df, ["img_spatial_centrality_score", "spatial_centrality_score"])
                i_ldr = _safe_mean(df, ["img_leadership_visual_context", "leadership_visual_context"])

                cnt_col = next((c for c in ["img_cross_image_appearance_count", "cross_image_appearance_count"] if c in df.columns), None)
                cnt_raw = round(float(df[cnt_col].dropna().mean()), 1) if cnt_col else 0.0
                i_cnt = round(min(cnt_raw * 10.0, 100.0), 1)

                img_data = pd.DataFrame({
                    "Métrica": [
                        "Reconocimiento Género (F) [w=0.20]",
                        "Plano de Importancia [w=0.10]",
                        "Ángulo de Cámara [w=0.10]",
                        "Regla de los Tercios [w=0.08]",
                        "Iluminación y Sombras [w=0.08]",
                        "Persistencia en Galería [w=0.10]",
                        "Centralidad Espacial [w=0.10]",
                        "Contexto de Liderazgo [w=0.14]",
                        "Aparición en Galería [w=0.10]",
                    ],
                    "Valor (%)": [i_gen, i_scl, i_ang, i_thd, i_shd, i_per, i_cen, i_ldr, i_cnt],
                    "TextoBarra": [f"{i_gen}%", f"{i_scl}%", f"{i_ang}%", f"{i_thd}%", f"{i_shd}%", f"{i_per}%", f"{i_cen}%", f"{i_ldr}%", f"{i_cnt}%"],
                })
                fig = px.bar(
                    img_data,
                    x="Valor (%)",
                    y="Métrica",
                    orientation="h",
                    text="TextoBarra",
                    range_x=[0, 110],
                    color_discrete_sequence=["#0284C7"],
                    title="Análisis de Imagen: Teoría Fotográfica, Planos y Sombras",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(_style_plotly(fig), use_container_width=True)

                sum_img = mod.get(
                    "Imagen",
                    round(
                        0.20 * i_gen + 0.10 * i_scl + 0.10 * i_ang + 0.08 * i_thd + 0.08 * i_shd
                        + 0.10 * i_per + 0.10 * i_cen + 0.14 * i_ldr + 0.10 * i_cnt, 1
                    )
                )

                st.markdown("---")
                st.markdown("##### Fórmula para el cálculo de la puntuación en imágenes")
                st.latex(r"\text{Score}_{\text{Imagen}} = \sum_{i=1}^{n} w_i \cdot \text{Métrica}_i \quad \text{donde} \quad \sum w_i = 1.0")
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px 16px; font-family: 'Montserrat', sans-serif; font-size: 0.85rem; color: #1E293B; margin-bottom: 1.5rem;">
                        <strong>Suma de componentes:</strong><br>
                        <code>(0.20 × {i_gen}) + (0.14 × {i_ldr}) + 0.10 × ({i_scl} + {i_ang} + {i_per} + {i_cen} + {i_cnt}) + 0.08 × ({i_thd} + {i_shd})</code><br>
                        = <strong style="color: #0A365C; font-size: 0.95rem;">{sum_img} / 100</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------------------
            # 4. MÉTRICAS DE TEXTO
            # -------------------------------------------------------------
            elif selected_chart == "4. Métricas de Texto y Léxico":
                t_fem = _safe_mean(df, ["txt_sentence_female_ratio", "sentence_female_ratio"])
                t_mal = _safe_mean(df, ["txt_sentence_male_ratio", "sentence_male_ratio"])
                t_neu = _safe_mean(df, ["txt_sentence_neutral_ratio", "sentence_neutral_ratio"])
                t_div = _safe_mean(df, ["txt_lexical_root_diversity", "lexical_root_diversity"])
                t_suf = _safe_mean(df, ["txt_feminine_suffix_ratio", "feminine_suffix_ratio"])
                t_frq = _safe_mean(df, ["txt_relative_frequency_score", "relative_frequency_score"])
                t_agn = _safe_mean(df, ["txt_syntactic_agency_ratio", "syntactic_agency_ratio"])
                t_exp = _safe_mean(df, ["txt_expert_quote_ratio", "expert_quote_ratio"])
                t_hed = _safe_mean(df, ["txt_headline_presence", "headline_presence"])

                cnt_col = next((c for c in ["txt_female_mention_frequency", "female_mention_frequency"] if c in df.columns), None)
                cnt_txt_raw = round(float(df[cnt_col].dropna().mean()), 1) if cnt_col else 0.0
                t_cnt = round(min((cnt_txt_raw / 15.0) * 100.0, 100.0), 1)

                txt_data = pd.DataFrame({
                    "Métrica": [
                        "Frases Referente Mujer [w=0.18]",
                        "Frases Referente Hombre [w=0.04]",
                        "Frases Neutras [w=0.06]",
                        "Diversidad Raíces Léxicas [w=0.08]",
                        "Terminaciones Femeninas [w=0.08]",
                        "Frecuencia Normalizada [w=0.10]",
                        "Agencia Sintáctica (Roles Activos) [w=0.16]",
                        "Citas como Experta [w=0.15]",
                        "Presencia en Titular [w=0.07]",
                        "Frecuencia Absoluta [w=0.08]",
                    ],
                    "Valor (%)": [t_fem, t_mal, t_neu, t_div, t_suf, t_frq, t_agn, t_exp, t_hed, t_cnt],
                    "TextoBarra": [f"{t_fem}%", f"{t_mal}%", f"{t_neu}%", f"{t_div}%", f"{t_suf}%", f"{t_frq}%", f"{t_agn}%", f"{t_exp}%", f"{t_hed}%", f"{t_cnt}%"],
                })
                fig = px.bar(
                    txt_data,
                    x="Valor (%)",
                    y="Métrica",
                    orientation="h",
                    text="TextoBarra",
                    range_x=[0, 110],
                    color_discrete_sequence=["#475569"],
                    title="Análisis de Texto: Clasificación de Frases, Raíces y Terminaciones",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(_style_plotly(fig), use_container_width=True)

                sum_txt = mod.get(
                    "Texto",
                    round(
                        0.18 * t_fem + 0.04 * t_mal + 0.06 * t_neu + 0.08 * t_div + 0.08 * t_suf
                        + 0.10 * t_frq + 0.16 * t_agn + 0.15 * t_exp + 0.07 * t_hed + 0.08 * t_cnt, 1
                    )
                )

                st.markdown("---")
                st.markdown("##### Fórmula para el cálculo de la puntuación en textos")
                st.latex(r"\text{Score}_{\text{Texto}} = \sum_{i=1}^{n} w_i \cdot \text{Métrica}_i \quad \text{donde} \quad \sum w_i = 1.0")
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px 16px; font-family: 'Montserrat', sans-serif; font-size: 0.85rem; color: #1E293B; margin-bottom: 1.5rem;">
                        <strong>Suma de componentes:</strong><br>
                        <code>(0.18 × {t_fem}) + (0.16 × {t_agn}) + (0.15 × {t_exp}) + (0.10 × {t_frq}) + 0.08 × ({t_div} + {t_suf} + {t_cnt}) + (0.07 × {t_hed}) + (0.06 × {t_neu}) + (0.04 × {t_mal})</code><br>
                        = <strong style="color: #0A365C; font-size: 0.95rem;">{sum_txt} / 100</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------------------
            # 5. MÉTRICAS DE AUDIO
            # -------------------------------------------------------------
            else:
                a_opn = _safe_mean(df, ["aud_female_opened_score", "female_opened_score"])
                a_mid = _safe_mean(df, ["aud_middle_segment_female_ratio", "middle_segment_female_ratio"])
                a_cls = _safe_mean(df, ["aud_female_closed_score", "female_closed_score"])
                a_tim = _safe_mean(df, ["aud_female_speaking_time_ratio", "female_speaking_time_ratio"])
                a_trn = _safe_mean(df, ["aud_speaking_turns_ratio", "speaking_turns_ratio"])
                a_cnd = _safe_mean(df, ["aud_acoustic_gender_confidence", "acoustic_gender_confidence"])
                a_noi = _safe_mean(df, ["aud_no_interruption_score", "no_interruption_score"])

                cnt_col = next((c for c in ["aud_female_turn_count", "female_turn_count"] if c in df.columns), None)
                cnt_aud_raw = round(float(df[cnt_col].dropna().mean()), 1) if cnt_col else 0.0
                a_cnt = round(min((cnt_aud_raw / 10.0) * 100.0, 100.0), 1)

                aud_data = pd.DataFrame({
                    "Métrica": [
                        "Apertura Femenina [w=0.10]",
                        "Presencia en Segmento Medio [w=0.12]",
                        "Cierre Femenino [w=0.10]",
                        "Tiempo Total Hablando [w=0.25]",
                        "Proporción Turnos de Habla [w=0.15]",
                        "Confianza Acústica Género [w=0.08]",
                        "Ausencia de Interrupciones [w=0.10]",
                        "Volumen de Intervenciones [w=0.10]",
                    ],
                    "Valor (%)": [a_opn, a_mid, a_cls, a_tim, a_trn, a_cnd, a_noi, a_cnt],
                    "TextoBarra": [f"{a_opn}%", f"{a_mid}%", f"{a_cls}%", f"{a_tim}%", f"{a_trn}%", f"{a_cnd}%", f"{a_noi}%", f"{a_cnt}%"],
                })
                fig = px.bar(
                    aud_data,
                    x="Valor (%)",
                    y="Métrica",
                    orientation="h",
                    text="TextoBarra",
                    range_x=[0, 110],
                    color_discrete_sequence=["#0EA5E9"],
                    title="Análisis de Audio: Apertura, Cierre, Diarización y Turnos",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(_style_plotly(fig), use_container_width=True)

                sum_aud = mod.get(
                    "Audio",
                    round(
                        0.10 * a_opn + 0.12 * a_mid + 0.10 * a_cls + 0.25 * a_tim
                        + 0.15 * a_trn + 0.08 * a_cnd + 0.10 * a_noi + 0.10 * a_cnt, 1
                    )
                )

                st.markdown("---")
                st.markdown("##### Fórmula para el cálculo de la puntuación en audios")
                st.latex(r"\text{Score}_{\text{Audio}} = \sum_{i=1}^{n} w_i \cdot \text{Métrica}_i \quad \text{donde} \quad \sum w_i = 1.0")
                st.markdown(
                    f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px 16px; font-family: 'Montserrat', sans-serif; font-size: 0.85rem; color: #1E293B; margin-bottom: 1.5rem;">
                        <strong>Suma de componentes:</strong><br>
                        <code>(0.25 × {a_tim}) + (0.15 × {a_trn}) + (0.12 × {a_mid}) + 0.10 × ({a_opn} + {a_cls} + {a_noi} + {a_cnt}) + (0.08 × {a_cnd})</code><br>
                        = <strong style="color: #0A365C; font-size: 0.95rem;">{sum_aud} / 100</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_esg_dashboard(company: str, score_data: dict):
    st.subheader(f"Ponderación Multimodal de Métricas: {company}")
    df = score_data.get("dataframe", pd.DataFrame())

    if df.empty:
        st.warning("No hay registros en el DataFrame para calcular métricas.")
        return

    if "by_modality" not in score_data:
        score_data["by_modality"] = {}
    mod = score_data["by_modality"]

    # 1. Medir la actividad real considerando las modalidades presentes por documento
    def get_doc_activity(row, mod_name: str, possible_cols: list[str], scale: float = 1.0) -> float:
        mods = row.get("available_modalities", [])
        if isinstance(mods, list) and mods and mod_name not in mods:
            return 0.0

        for col in possible_cols:
            if col in row and not pd.isna(row[col]):
                return float(row[col]) * scale
        return 0.0

    df_calc = df.copy()
    df_calc["act_Texto"] = df_calc.apply(
        lambda r: get_doc_activity(r, "Texto", ["txt_female_mention_frequency", "female_mention_frequency"], 1.0),
        axis=1,
    )
    df_calc["act_Audio"] = df_calc.apply(
        lambda r: get_doc_activity(r, "Audio", ["aud_female_turn_count", "female_turn_count"], 1.0),
        axis=1,
    )
    df_calc["act_Imagen"] = df_calc.apply(
        lambda r: get_doc_activity(r, "Imagen", ["img_cross_image_appearance_count", "cross_image_appearance_count"], 1.0),
        axis=1,
    )
    df_calc["act_Vídeo"] = df_calc.apply(
        lambda r: get_doc_activity(r, "Vídeo", ["vid_female_screen_time_ratio", "female_screen_time_ratio"], 10.0),
        axis=1,
    )

    # 2. Sincronización canónica de los scores por modalidad (tarjeta superior == desglose)
    t_fem = _safe_mean(df, ["txt_sentence_female_ratio", "sentence_female_ratio"])
    t_mal = _safe_mean(df, ["txt_sentence_male_ratio", "sentence_male_ratio"])
    t_neu = _safe_mean(df, ["txt_sentence_neutral_ratio", "sentence_neutral_ratio"])
    t_div = _safe_mean(df, ["txt_lexical_root_diversity", "lexical_root_diversity"])
    t_suf = _safe_mean(df, ["txt_feminine_suffix_ratio", "feminine_suffix_ratio"])
    t_frq = _safe_mean(df, ["txt_relative_frequency_score", "relative_frequency_score"])
    t_agn = _safe_mean(df, ["txt_syntactic_agency_ratio", "syntactic_agency_ratio"])
    t_exp = _safe_mean(df, ["txt_expert_quote_ratio", "expert_quote_ratio"])
    t_hed = _safe_mean(df, ["txt_headline_presence", "headline_presence"])
    cnt_txt_col = next((c for c in ["txt_female_mention_frequency", "female_mention_frequency"] if c in df.columns), None)
    cnt_txt_raw = round(float(df[cnt_txt_col].dropna().mean()), 1) if cnt_txt_col else 0.0
    t_cnt = round(min((cnt_txt_raw / 15.0) * 100.0, 100.0), 1)

    mod["Texto"] = round(
        0.18 * t_fem + 0.04 * t_mal + 0.06 * t_neu + 0.08 * t_div + 0.08 * t_suf
        + 0.10 * t_frq + 0.16 * t_agn + 0.15 * t_exp + 0.07 * t_hed + 0.08 * t_cnt,
        1,
    )

    a_opn = _safe_mean(df, ["aud_female_opened_score", "female_opened_score"])
    a_mid = _safe_mean(df, ["aud_middle_segment_female_ratio", "middle_segment_female_ratio"])
    a_cls = _safe_mean(df, ["aud_female_closed_score", "female_closed_score"])
    a_tim = _safe_mean(df, ["aud_female_speaking_time_ratio", "female_speaking_time_ratio"])
    a_trn = _safe_mean(df, ["aud_speaking_turns_ratio", "speaking_turns_ratio"])
    a_cnd = _safe_mean(df, ["aud_acoustic_gender_confidence", "acoustic_gender_confidence"])
    a_noi = _safe_mean(df, ["aud_no_interruption_score", "no_interruption_score"])
    cnt_aud_col = next((c for c in ["aud_female_turn_count", "female_turn_count"] if c in df.columns), None)
    cnt_aud_raw = round(float(df[cnt_aud_col].dropna().mean()), 1) if cnt_aud_col else 0.0
    a_cnt = round(min((cnt_aud_raw / 10.0) * 100.0, 100.0), 1)

    mod["Audio"] = round(
        0.10 * a_opn + 0.12 * a_mid + 0.10 * a_cls + 0.25 * a_tim
        + 0.15 * a_trn + 0.08 * a_cnd + 0.10 * a_noi + 0.10 * a_cnt,
        1,
    )

    i_gen = _safe_mean(df, ["img_visual_gender_female_ratio", "visual_gender_female_ratio"])
    i_scl = _safe_mean(df, ["img_shot_scale_importance_score", "shot_scale_importance_score"])
    i_ang = _safe_mean(df, ["img_camera_angle_power_score", "camera_angle_power_score"])
    i_thd = _safe_mean(df, ["img_rule_of_thirds_alignment", "rule_of_thirds_alignment"])
    i_shd = _safe_mean(df, ["img_shadow_lighting_prominence", "shadow_lighting_prominence"])
    i_per = _safe_mean(df, ["img_cross_image_persistence_ratio", "cross_image_persistence_ratio"])
    i_cen = _safe_mean(df, ["img_spatial_centrality_score", "spatial_centrality_score"])
    i_ldr = _safe_mean(df, ["img_leadership_visual_context", "leadership_visual_context"])
    cnt_img_col = next((c for c in ["img_cross_image_appearance_count", "cross_image_appearance_count"] if c in df.columns), None)
    cnt_img_raw = round(float(df[cnt_img_col].dropna().mean()), 1) if cnt_img_col else 0.0
    i_cnt = round(min(cnt_img_raw * 10.0, 100.0), 1)

    mod["Imagen"] = round(
        0.20 * i_gen + 0.10 * i_scl + 0.10 * i_ang + 0.08 * i_thd + 0.08 * i_shd
        + 0.10 * i_per + 0.10 * i_cen + 0.14 * i_ldr + 0.10 * i_cnt,
        1,
    )

    v_opt = _safe_mean(df, ["vid_optical_flow_motion_score", "optical_flow_motion_score"])
    v_kin = _safe_mean(df, ["vid_kinetic_dominance_ratio", "kinetic_dominance_ratio"])
    v_sil = _safe_mean(df, ["vid_silhouette_change_histogram_score", "silhouette_change_histogram_score"])
    v_scr = _safe_mean(df, ["vid_female_screen_time_ratio", "female_screen_time_ratio"])
    v_spk = _safe_mean(df, ["vid_active_speaker_sync_score", "active_speaker_sync_score"])

    mod["Vídeo"] = round(
        0.15 * v_opt + 0.15 * v_kin + 0.15 * v_sil + 0.35 * v_scr + 0.20 * v_spk,
        1,
    )

    # 3. Pesos normalizados de cada canal
    weights = score_data.get("modality_weights")
    if not weights:
        sum_txt = df_calc["act_Texto"].sum()
        sum_aud = df_calc["act_Audio"].sum()
        sum_img = df_calc["act_Imagen"].sum()
        sum_vid = df_calc["act_Vídeo"].sum()

        total_act = max(sum_txt + sum_aud + sum_img + sum_vid, 1.0)
        weights = {
            "Texto": sum_txt / total_act,
            "Audio": sum_aud / total_act,
            "Imagen": sum_img / total_act,
            "Vídeo": sum_vid / total_act,
        }

    # 4. Índice global
    _, col_global, _ = st.columns([1, 1.8, 1])
    with col_global:
        st.markdown(
            f"""
            <div style="
                background: #FFFFFF; border: 2.5px solid #0A365C; border-radius: 12px;
                padding: 18px 20px; text-align: center; box-shadow: 0 4px 14px rgba(10, 54, 92, 0.12);
                margin-bottom: 0.5rem;
            ">
                <div style="font-family: 'Montserrat', sans-serif; font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.8px;">
                    Índice Global Ponderado
                </div>
                <div style="font-family: 'Merriweather', serif; font-size: 2.4rem; font-weight: 700; color: #0F172A; margin-top: 4px;">
                    {score_data.get('global_score', 0.0)} <span style="font-size: 1rem; color: #94A3B8; font-weight: 400;">/ 100</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    modalities = [
        ("Texto", mod.get("Texto", 0.0)),
        ("Audio", mod.get("Audio", 0.0)),
        ("Imagen", mod.get("Imagen", 0.0)),
        ("Vídeo", mod.get("Vídeo", 0.0)),
    ]

    arrow_cols = st.columns(4)
    for col, (name, _) in zip(arrow_cols, modalities):
        w_pct = round(weights.get(name, 0.25) * 100, 1)
        with col:
            st.markdown(
                f"""
                <div style="text-align: center; margin: 4px 0 8px 0;">
                    <div style="display: inline-block; background-color: #F1F5F9; border: 1px solid #94A3B8; color: #1E293B; font-family: 'Montserrat', sans-serif; font-size: 0.78rem; font-weight: 700; padding: 3px 10px; border-radius: 6px;">
                        Peso: {w_pct}%
                    </div>
                    <div style="color: #0A365C; font-size: 1.35rem; line-height: 1; margin-top: 3px; font-weight: 900;">↓</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    card_cols = st.columns(4)
    for col, (name, val) in zip(card_cols, modalities):
        with col:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; text-align: center; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);">
                    <div style="font-family: 'Montserrat', sans-serif; font-size: 0.8rem; font-weight: 600; color: #64748B;">
                        Score {name}
                    </div>
                    <div style="font-family: 'Merriweather', serif; font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-top: 4px;">
                        {val} <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 400;">/ 100</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    _render_interactive_charts(company, score_data, df, df_calc, weights)