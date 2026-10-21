from typing import Any, Dict, List
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.common.schemas import ExtractedFeaturesPayload

FEATURE_COLUMNS = [
    # Texto (9)
    "txt_sentence_female_ratio",
    "txt_sentence_male_ratio",
    "txt_sentence_neutral_ratio",
    "txt_lexical_root_diversity",
    "txt_feminine_suffix_ratio",
    "txt_relative_frequency_score",
    "txt_syntactic_agency_ratio",
    "txt_expert_quote_ratio",
    "txt_headline_presence",
    # Audio (7)
    "aud_female_opened_score",
    "aud_female_closed_score",
    "aud_middle_segment_female_ratio",
    "aud_female_speaking_time_ratio",
    "aud_speaking_turns_ratio",
    "aud_acoustic_gender_confidence",
    "aud_no_interruption_score",
    # Imagen (8)
    "img_visual_gender_female_ratio",
    "img_shot_scale_importance_score",
    "img_camera_angle_power_score",
    "img_rule_of_thirds_alignment",
    "img_shadow_lighting_prominence",
    "img_cross_image_persistence_ratio",
    "img_spatial_centrality_score",
    "img_leadership_visual_context",
    # Vídeo (5)
    "vid_optical_flow_motion_score",
    "vid_kinetic_dominance_ratio",
    "vid_silhouette_change_histogram_score",
    "vid_female_screen_time_ratio",
    "vid_active_speaker_sync_score",
]


class CorporateGenderScorer:

    def __init__(self):
        self.model = None
        self._train_calibrated_baseline()

    def _train_calibrated_baseline(self):
        """Entrena el modelo LightGBM sobre las 29 métricas multimodales."""
        rng = np.random.default_rng(42)
        X_syn = pd.DataFrame(
            rng.uniform(0, 1, size=(800, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )

        # Pesos calibrados para negocio ESG (suman ~100 en contribuciones positivas)
        weights = np.array([
            # Texto (28 pts positivos, -4 penalización sesgo masculino)
            4.0,
            -4.0,
            1.0,
            3.0,
            3.0,
            3.0,
            5.0,
            5.0,
            4.0,
            # Audio (24 pts)
            3.5,
            3.5,
            3.0,
            5.0,
            3.5,
            2.0,
            3.5,
            # Imagen (24 pts)
            3.5,
            4.0,
            2.5,
            2.0,
            3.0,
            3.0,
            3.0,
            3.0,
            # Vídeo (24 pts)
            5.0,
            4.5,
            4.0,
            5.5,
            5.0,
        ])

        y_syn = np.clip(
            X_syn.values @ weights + rng.normal(0, 1.5, 800), 0, 100
        )

        # Restricciones monótonas: +1 excepto sentence_male_ratio (-1) y sentence_neutral_ratio (0)
        monotone_constraints = [
            1,
            -1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]

        self.model = lgb.LGBMRegressor(
            n_estimators=90,
            learning_rate=0.07,
            monotone_constraints=monotone_constraints,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_syn, y_syn)

    def build_feature_matrix(
        self, docs: List[ExtractedFeaturesPayload]
    ) -> pd.DataFrame:
        rows = [d.to_feature_dict() for d in docs]
        return pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

    def _calc_text_score(self, tm: Any) -> float:
        if tm is None:
            return np.nan
        cnt_norm = min(float(tm.female_mention_frequency) / 15.0, 1.0)
        score = (
            0.18 * tm.sentence_female_ratio
            + 0.04 * tm.sentence_male_ratio
            + 0.06 * tm.sentence_neutral_ratio
            + 0.08 * tm.lexical_root_diversity
            + 0.08 * tm.feminine_suffix_ratio
            + 0.10 * tm.relative_frequency_score
            + 0.16 * tm.syntactic_agency_ratio
            + 0.15 * tm.expert_quote_ratio
            + 0.07 * tm.headline_presence
            + 0.08 * cnt_norm
        ) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def _calc_audio_score(self, am: Any) -> float:
        if am is None:
            return np.nan
        cnt_norm = min(float(am.female_turn_count) / 10.0, 1.0)
        score = (
            0.10 * am.female_opened_score
            + 0.12 * am.middle_segment_female_ratio
            + 0.10 * am.female_closed_score
            + 0.25 * am.female_speaking_time_ratio
            + 0.15 * am.speaking_turns_ratio
            + 0.08 * am.acoustic_gender_confidence
            + 0.10 * am.no_interruption_score
            + 0.10 * cnt_norm
        ) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def _calc_image_score(self, im: Any) -> float:
        if im is None:
            return np.nan
        cnt_norm = min(float(im.cross_image_appearance_count) / 10.0, 1.0)
        score = (
            0.20 * im.visual_gender_female_ratio
            + 0.10 * im.shot_scale_importance_score
            + 0.10 * im.camera_angle_power_score
            + 0.08 * im.rule_of_thirds_alignment
            + 0.08 * im.shadow_lighting_prominence
            + 0.10 * im.cross_image_persistence_ratio
            + 0.10 * im.spatial_centrality_score
            + 0.14 * im.leadership_visual_context
            + 0.10 * cnt_norm
        ) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def _calc_video_score(self, vm: Any) -> float:
        if vm is None:
            return np.nan
        score = (
            0.15 * vm.optical_flow_motion_score
            + 0.15 * vm.kinetic_dominance_ratio
            + 0.15 * vm.silhouette_change_histogram_score
            + 0.35 * vm.female_screen_time_ratio
            + 0.20 * vm.active_speaker_sync_score
        ) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def predict_importance_score(
        self, docs: List[ExtractedFeaturesPayload]
    ) -> Dict[str, Any]:
        features_list = [d.to_feature_dict() for d in docs]
        features_df = pd.DataFrame(features_list)

        # 1. Alinear exactamente las 29 columnas con las de entrenamiento para LightGBM y SHAP
        features_for_model = features_df.reindex(
            columns=FEATURE_COLUMNS, fill_value=0.0
        ).fillna(0.0)

        if hasattr(self.model, "predict"):
            base_preds = self.model.predict(features_for_model)
        else:
            base_preds = [50.0] * len(docs)

        # 2. DataFrame resultante completo (conserva todas las columnas del schema + metadatos)
        df_results = features_df.copy()
        df_results["doc_id"] = [d.doc_id for d in docs]
        df_results["source"] = [
            getattr(d.source, "value", str(d.source)) for d in docs
        ]
        df_results["title"] = [d.title for d in docs]
        df_results["demographic_subgroup"] = [
            d.demographic_subgroup for d in docs
        ]
        df_results["available_modalities"] = [
            d.available_modalities for d in docs
        ]

        # 3. Extracción segura considerando objetos de audio globales o individuales
        df_results["aud_opened_by"] = [
            getattr(
                d.audio_metrics.opened_by, "value", str(d.audio_metrics.opened_by)
            )
            if getattr(d, "audio_metrics", None) is not None
            and getattr(d.audio_metrics, "opened_by", None) is not None
            else None
            for d in docs
        ]
        df_results["aud_closed_by"] = [
            getattr(
                d.audio_metrics.closed_by, "value", str(d.audio_metrics.closed_by)
            )
            if getattr(d, "audio_metrics", None) is not None
            and getattr(d.audio_metrics, "closed_by", None) is not None
            else None
            for d in docs
        ]

        # 4. Puntuaciones directas normalizadas
        scores_texto = [self._calc_text_score(d.text_metrics) for d in docs]
        scores_audio = [self._calc_audio_score(d.audio_metrics) for d in docs]
        scores_imagen = [self._calc_image_score(d.image_metrics) for d in docs]
        scores_video = [self._calc_video_score(d.video_metrics) for d in docs]

        df_results["score_texto"] = scores_texto
        df_results["score_audio"] = scores_audio
        df_results["score_imagen"] = scores_imagen
        df_results["score_video"] = scores_video

        # 5. Score final por documento: media balanceada de las modalidades activas
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

            score_val = (
                np.mean(valid_scores) if valid_scores else float(base_preds[i])
            )
            doc_final_scores.append(round(float(score_val), 1))

        df_results["importance_score"] = doc_final_scores

        by_modality = {
            "Texto": (
                round(float(np.nanmean(scores_texto)), 1)
                if not np.all(np.isnan(scores_texto))
                else 0.0
            ),
            "Audio": (
                round(float(np.nanmean(scores_audio)), 1)
                if not np.all(np.isnan(scores_audio))
                else 0.0
            ),
            "Imagen": (
                round(float(np.nanmean(scores_imagen)), 1)
                if not np.all(np.isnan(scores_imagen))
                else 0.0
            ),
            "Vídeo": (
                round(float(np.nanmean(scores_video)), 1)
                if not np.all(np.isnan(scores_video))
                else 0.0
            ),
        }

        return {
            "global_score": round(float(np.mean(doc_final_scores)), 1),
            "by_modality": by_modality,
            "dataframe": df_results,
            "features_only": features_for_model,
        }