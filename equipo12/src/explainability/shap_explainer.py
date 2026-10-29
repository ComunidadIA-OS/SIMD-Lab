import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FEATURE_LABELS = {
    # Texto
    "txt_sentence_female_ratio": "Texto: Frases con Referente Mujer",
    "txt_sentence_male_ratio": "Texto: Frases con Referente Hombre",
    "txt_sentence_neutral_ratio": "Texto: Frases Neutras",
    "txt_lexical_root_diversity": "Texto: Diversidad de Raíces Léxicas",
    "txt_feminine_suffix_ratio": "Texto: Terminaciones Femeninas",
    "txt_relative_frequency_score": "Texto: Frecuencia Relativa",
    "txt_syntactic_agency_ratio": "Texto: Agencia Sintáctica (Roles Activos)",
    "txt_expert_quote_ratio": "Texto: Citas como Experta",
    "txt_headline_presence": "Texto: Presencia en Titular",
    "txt_female_mention_frequency": "Texto: Frecuencia de Menciones",
    # Vídeo
    "vid_optical_flow_motion_score": "Vídeo: Flujo Óptico (Movimiento)",
    "vid_kinetic_dominance_ratio": "Vídeo: Dominancia Cinética",
    "vid_silhouette_change_histogram_score": "Vídeo: Histograma de Silueta",
    "vid_female_screen_time_ratio": "Vídeo: Tiempo en Pantalla",
    "vid_active_speaker_sync_score": "Vídeo: Sincronía Labio-Voz",
    # Audio
    "aud_female_opened_score": "Audio: Apertura de Habla",
    "aud_middle_segment_female_ratio": "Audio: Presencia Segmento Central",
    "aud_female_closed_score": "Audio: Cierre de Habla",
    "aud_female_speaking_time_ratio": "Audio: Tiempo de Habla Total",
    "aud_speaking_turns_ratio": "Audio: Proporción Turnos de Habla",
    "aud_acoustic_gender_confidence": "Audio: Confianza Acústica de Género",
    "aud_no_interruption_score": "Audio: Ausencia de Interrupciones",
    "aud_female_turn_count": "Audio: Conteo de Intervenciones",
    # Imagen
    "img_visual_gender_female_ratio": "Imagen: Reconocimiento Facial Femenino",
    "img_shot_scale_importance_score": "Imagen: Escala del Plano de Enfoque",
    "img_camera_angle_power_score": "Imagen: Ángulo de Cámara (Poder)",
    "img_rule_of_thirds_alignment": "Imagen: Regla de los Tercios",
    "img_shadow_lighting_prominence": "Imagen: Iluminación y Sombras",
    "img_cross_image_persistence_ratio": "Imagen: Persistencia en Galería",
    "img_spatial_centrality_score": "Imagen: Centralidad Espacial",
    "img_leadership_visual_context": "Imagen: Contexto de Liderazgo",
    "img_cross_image_appearance_count": "Imagen: Frecuencia en Imágenes",
}


def _traducir_textos_shap(fig):
  """Reemplaza textos técnicos en inglés y fórmulas matemáticas por términos en español."""
  for ax in fig.get_axes():
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    new_labels = []
    modificado = False
    for lbl in labels:
      if "Sum of" in lbl and "other features" in lbl:
        num = "".join(filter(str.isdigit, lbl))
        new_labels.append(f"Suma de {num} variables restantes")
        modificado = True
      else:
        new_labels.append(lbl)
    if modificado:
      ax.set_yticklabels(new_labels)

    for txt in ax.texts:
      t = txt.get_text()
      if "Sum of" in t and "other features" in t:
        num = "".join(filter(str.isdigit, t))
        txt.set_text(f"Suma de {num} variables restantes")
      elif "mean(|SHAP value|)" in t:
        txt.set_text("Impacto medio (|valor SHAP|)")
      elif "$f(x)$" in t or "f(x)" in t:
        val = t.replace("$f(x)$", "").replace("f(x)", "").strip(" =:")
        txt.set_text(f"Score asignado: {val}")
      elif "E[f(x)]" in t:
        val = t.replace("E[f(x)]", "").strip(" =:")
        txt.set_text(f"Punto neutro paridad: {val}")

    if ax.get_xlabel() == "mean(|SHAP value|)":
      ax.set_xlabel("Impacto medio (|valor SHAP|)")


class ShapExplanationEngine:

    def __init__(self, scorer):
        self.scorer = scorer
        self.weights = scorer.weights
        self.base_parity_value = 0.5  # Referencia de paridad/neutralidad (50%)

    def _compute_exact_shap(self, features_df: pd.DataFrame):
        """Calcula los valores exactos de Shapley analíticamente sobre la ponderación real."""
        cols = features_df.columns.tolist()
        vals_matrix = np.zeros(features_df.shape)
        data_matrix = np.nan_to_num(features_df.values, nan=0.0)

        for j, col in enumerate(cols):
            w = self.weights.get(col, 0.0) * 100.0  # Escala sobre 100
            x = features_df[col].values

            # Penalización en caso de frases masculinas predominantes
            if col == "txt_sentence_male_ratio":
                phi = -w * (x - self.base_parity_value)
            else:
                phi = w * (x - self.base_parity_value)

            # Si el documento no tiene esta métrica (NaN), el impacto de Shapley es exactamente 0
            phi = np.where(np.isnan(x), 0.0, phi)
            vals_matrix[:, j] = phi

        base_val = 50.0  # Base neutral E[f(x)]
        return vals_matrix, data_matrix, base_val

    def plot_global_summary(
        self, features_df: pd.DataFrame, max_display: int = 30
    ):
        """Calcula el impacto global de variables con etiquetas en español y sin agrupar."""
        vals_matrix, data_matrix, base_val = self._compute_exact_shap(features_df)
        renamed_features = [
            FEATURE_LABELS.get(c, c) for c in features_df.columns.tolist()
        ]
        clean_exp = shap.Explanation(
            values=vals_matrix,
            base_values=np.full(features_df.shape[0], base_val),
            data=data_matrix,
            feature_names=renamed_features,
        )

        n_features = len(features_df.columns)
        display_limit = max(max_display, n_features + 1)

        fig, ax = plt.subplots(figsize=(9, 8))
        shap.plots.bar(clean_exp, max_display=display_limit, show=False)
        _traducir_textos_shap(fig)

        plt.title(
            "Impacto Global de Variables en el Score Multimodal (SHAP Analítico)",
            fontsize=13,
            fontfamily="serif",
            color="#0F172A",
        )
        plt.tight_layout()
        return fig

    def plot_local_waterfall(
        self,
        features_df: pd.DataFrame,
        doc_index: int = 0,
        modality_prefix: str = None,
    ):
        vals_matrix, data_matrix, base_val = self._compute_exact_shap(features_df)
        all_cols = features_df.columns.tolist()

        doc_vals = vals_matrix[doc_index]
        doc_data = data_matrix[doc_index]

        if modality_prefix:
            indices = [i for i, c in enumerate(all_cols) if c.startswith(modality_prefix)]
            raw_doc_slice = features_df.iloc[doc_index, indices]

            # Comprobar si todas las métricas de esa modalidad son NaN o están ausentes
            if raw_doc_slice.isna().all() or np.all(doc_data[indices] == 0.0):
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.text(
                    0.5, 0.5,
                    "Modalidad no detectada en este documento.\nSin impacto algorítmico asignado.",
                    horizontalalignment="center", verticalalignment="center",
                    fontsize=12, color="#64748B"
                )
                ax.axis("off")
                return fig

            doc_vals = doc_vals[indices]
            doc_data = doc_data[indices]
            filtered_cols = [all_cols[i] for i in indices]
        else:
            filtered_cols = all_cols

        renamed_features = [FEATURE_LABELS.get(c, c) for c in filtered_cols]

        clean_doc_exp = shap.Explanation(
            values=doc_vals,
            base_values=base_val,
            data=doc_data,
            feature_names=renamed_features,
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(
            clean_doc_exp, max_display=len(renamed_features) + 1, show=False
        )
        _traducir_textos_shap(fig)
        plt.tight_layout()
        return fig