from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.common.schemas import ExtractedFeaturesPayload

# Pesos oficiales ponderados de cada métrica (suman 1.0 en cada modalidad)
PESOS_MODALIDAD = {
    # Texto
    "txt_sentence_female_ratio": 0.18,
    "txt_sentence_male_ratio": 0.04,
    "txt_sentence_neutral_ratio": 0.06,
    "txt_lexical_root_diversity": 0.08,
    "txt_feminine_suffix_ratio": 0.08,
    "txt_relative_frequency_score": 0.10,
    "txt_syntactic_agency_ratio": 0.16,
    "txt_expert_quote_ratio": 0.15,
    "txt_headline_presence": 0.07,
    "txt_female_mention_frequency": 0.08,
    # Audio
    "aud_female_opened_score": 0.10,
    "aud_middle_segment_female_ratio": 0.12,
    "aud_female_closed_score": 0.10,
    "aud_female_speaking_time_ratio": 0.25,
    "aud_speaking_turns_ratio": 0.15,
    "aud_acoustic_gender_confidence": 0.08,
    "aud_no_interruption_score": 0.10,
    "aud_female_turn_count": 0.10,
    # Imagen
    "img_visual_gender_female_ratio": 0.20,
    "img_shot_scale_importance_score": 0.10,
    "img_camera_angle_power_score": 0.10,
    "img_rule_of_thirds_alignment": 0.08,
    "img_shadow_lighting_prominence": 0.08,
    "img_cross_image_persistence_ratio": 0.10,
    "img_spatial_centrality_score": 0.10,
    "img_leadership_visual_context": 0.14,
    "img_cross_image_appearance_count": 0.10,
    # Vídeo
    "vid_optical_flow_motion_score": 0.15,
    "vid_kinetic_dominance_ratio": 0.15,
    "vid_silhouette_change_histogram_score": 0.15,
    "vid_female_screen_time_ratio": 0.35,
    "vid_active_speaker_sync_score": 0.20,
}

FEATURE_COLUMNS = list(PESOS_MODALIDAD.keys())


def _ponderada(pares) -> float:
    """Media ponderada (0-100) estrictamente de las métricas que tienen dato medido."""
    validos = [(w, float(v)) for w, v in pares if v is not None and not np.isnan(v)]
    if not validos:
        return np.nan
    peso = sum(w for w, _ in validos)
    return float(np.clip(sum(w * v for w, v in validos) / peso * 100.0, 0.0, 100.0))


class CorporateGenderScorer:

    def __init__(self):
        # Eliminada la dependencia de LightGBM y su entrenamiento sintético
        self.weights = PESOS_MODALIDAD

    def build_feature_matrix(
        self, docs: List[ExtractedFeaturesPayload]
    ) -> pd.DataFrame:
        rows = [d.to_feature_dict() for d in docs]
        return pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS).reset_index(drop=True)

    def _calc_text_score(self, tm: Any) -> float:
        if tm is None:
            return np.nan
        cnt_norm = min(float(tm.female_mention_frequency) / 15.0, 1.0)
        return _ponderada([
            (self.weights["txt_sentence_female_ratio"], tm.sentence_female_ratio),
            (self.weights["txt_sentence_male_ratio"], tm.sentence_male_ratio),
            (self.weights["txt_sentence_neutral_ratio"], tm.sentence_neutral_ratio),
            (self.weights["txt_lexical_root_diversity"], tm.lexical_root_diversity),
            (self.weights["txt_feminine_suffix_ratio"], tm.feminine_suffix_ratio),
            (self.weights["txt_relative_frequency_score"], tm.relative_frequency_score),
            (self.weights["txt_syntactic_agency_ratio"], tm.syntactic_agency_ratio),
            (self.weights["txt_expert_quote_ratio"], tm.expert_quote_ratio),
            (self.weights["txt_headline_presence"], tm.headline_presence),
            (self.weights["txt_female_mention_frequency"], cnt_norm),
        ])

    def _calc_audio_score(self, am: Any) -> float:
        if am is None:
            return np.nan
        cnt_norm = min(float(am.female_turn_count) / 10.0, 1.0)
        return _ponderada([
            (self.weights["aud_female_opened_score"], am.female_opened_score),
            (self.weights["aud_middle_segment_female_ratio"], am.middle_segment_female_ratio),
            (self.weights["aud_female_closed_score"], am.female_closed_score),
            (self.weights["aud_female_speaking_time_ratio"], am.female_speaking_time_ratio),
            (self.weights["aud_speaking_turns_ratio"], am.speaking_turns_ratio),
            (self.weights["aud_acoustic_gender_confidence"], am.acoustic_gender_confidence),
            (self.weights["aud_no_interruption_score"], am.no_interruption_score),
            (self.weights["aud_female_turn_count"], cnt_norm),
        ])

    def _calc_image_score(self, im: Any) -> float:
        if im is None:
            return np.nan
        cnt_norm = min(float(im.cross_image_appearance_count) / 10.0, 1.0)
        return _ponderada([
            (self.weights["img_visual_gender_female_ratio"], im.visual_gender_female_ratio),
            (self.weights["img_shot_scale_importance_score"], im.shot_scale_importance_score),
            (self.weights["img_camera_angle_power_score"], im.camera_angle_power_score),
            (self.weights["img_rule_of_thirds_alignment"], im.rule_of_thirds_alignment),
            (self.weights["img_shadow_lighting_prominence"], im.shadow_lighting_prominence),
            (self.weights["img_cross_image_persistence_ratio"], im.cross_image_persistence_ratio),
            (self.weights["img_spatial_centrality_score"], im.spatial_centrality_score),
            (self.weights["img_leadership_visual_context"], im.leadership_visual_context),
            (self.weights["img_cross_image_appearance_count"], cnt_norm),
        ])

    def _calc_video_score(self, vm: Any) -> float:
        if vm is None:
            return np.nan
        return _ponderada([
            (self.weights["vid_optical_flow_motion_score"], vm.optical_flow_motion_score),
            (self.weights["vid_kinetic_dominance_ratio"], vm.kinetic_dominance_ratio),
            (self.weights["vid_silhouette_change_histogram_score"], vm.silhouette_change_histogram_score),
            (self.weights["vid_female_screen_time_ratio"], vm.female_screen_time_ratio),
            (self.weights["vid_active_speaker_sync_score"], vm.active_speaker_sync_score),
        ])

    def predict_importance_score(
        self, docs: List[ExtractedFeaturesPayload]
    ) -> Dict[str, Any]:
        features_list = [d.to_feature_dict() for d in docs]
        features_df = pd.DataFrame(features_list)

        # Matriz para SHAP: se preservan estrictamente los NaN originales sin inventar datos
        features_for_model = features_df.reindex(columns=FEATURE_COLUMNS).reset_index(drop=True)

        # DataFrame resultante completo
        df_results = features_df.copy()
        df_results["doc_id"] = [d.doc_id for d in docs]
        df_results["source"] = [getattr(d.source, "value", str(d.source)) for d in docs]
        df_results["title"] = [d.title for d in docs]
        df_results["demographic_subgroup"] = [d.demographic_subgroup for d in docs]
        df_results["available_modalities"] = [d.available_modalities for d in docs]

        df_results["aud_opened_by"] = [
            getattr(d.audio_metrics.opened_by, "value", str(d.audio_metrics.opened_by))
            if getattr(d, "audio_metrics", None) is not None
            and getattr(d.audio_metrics, "opened_by", None) is not None
            else None
            for d in docs
        ]
        df_results["aud_closed_by"] = [
            getattr(d.audio_metrics.closed_by, "value", str(d.audio_metrics.closed_by))
            if getattr(d, "audio_metrics", None) is not None
            and getattr(d.audio_metrics, "closed_by", None) is not None
            else None
            for d in docs
        ]

        # Puntuaciones directas normalizadas
        scores_texto = [self._calc_text_score(d.text_metrics) for d in docs]
        scores_audio = [self._calc_audio_score(d.audio_metrics) for d in docs]
        scores_imagen = [self._calc_image_score(d.image_metrics) for d in docs]
        scores_video = [self._calc_video_score(d.video_metrics) for d in docs]

        df_results["score_texto"] = scores_texto
        df_results["score_audio"] = scores_audio
        df_results["score_imagen"] = scores_imagen
        df_results["score_video"] = scores_video

        # Puntuación final por documento: media simple de sus modalidades presentes
        doc_final_scores = []
        for i, d in enumerate(docs):
            mods = d.available_modalities
            valid_scores = []
            if "Texto" in mods and not np.isnan(scores_texto[i]):
                valid_scores.append(scores_texto[i])
            if "Audio" in mods and not np.isnan(scores_audio[i]):
                valid_scores.append(scores_audio[i])
            if "Imagen" in mods and not np.isnan(scores_imagen[i]):
                valid_scores.append(scores_imagen[i])
            if "Vídeo" in mods and not np.isnan(scores_video[i]):
                valid_scores.append(scores_video[i])

            score_val = np.mean(valid_scores) if valid_scores else np.nan
            doc_final_scores.append(round(float(score_val), 1) if not np.isnan(score_val) else np.nan)

        df_results["importance_score"] = doc_final_scores

        by_modality = {
            "Texto": (round(float(np.nanmean(scores_texto)), 1) if not np.all(np.isnan(scores_texto)) else None),
            "Audio": (round(float(np.nanmean(scores_audio)), 1) if not np.all(np.isnan(scores_audio)) else None),
            "Imagen": (round(float(np.nanmean(scores_imagen)), 1) if not np.all(np.isnan(scores_imagen)) else None),
            "Vídeo": (round(float(np.nanmean(scores_video)), 1) if not np.all(np.isnan(scores_video)) else None),
        }

        # Ponderación del Índice Global por presencia efectiva de cada modalidad
        modalidades_valores = [
            ("Texto", by_modality.get("Texto"), sum(1 for s in scores_texto if not np.isnan(s))),
            ("Audio", by_modality.get("Audio"), sum(1 for s in scores_audio if not np.isnan(s))),
            ("Imagen", by_modality.get("Imagen"), sum(1 for s in scores_imagen if not np.isnan(s))),
            ("Vídeo", by_modality.get("Vídeo"), sum(1 for s in scores_video if not np.isnan(s))),
        ]

        valid_mods = [(peso, val) for _, val, peso in modalidades_valores if val is not None and peso > 0]
        
        if valid_mods:
            peso_total = sum(p for p, _ in valid_mods)
            global_ponderado = round(sum(p * v for p, v in valid_mods) / peso_total, 1)
        else:
            valid_docs = [s for s in doc_final_scores if not np.isnan(s)]
            global_ponderado = round(float(np.mean(valid_docs)), 1) if valid_docs else 0.0

        return {
            "global_score": global_ponderado,
            "by_modality": by_modality,
            "dataframe": df_results,
            "features_only": features_for_model,
        }