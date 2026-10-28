import json
import math
from pathlib import Path
import struct
from typing import List, Optional
import wave
from PIL import Image, ImageDraw

from src.common.schemas import (
    AudioMetrics,
    AudioSample,
    ExtractedFeaturesPayload,
    GenderTarget,
    ImageMetrics,
    ImageSample,
    MediaSource,
    TextMetrics,
    TextSample,
    VideoMetrics,
    VideoSample,
)
from src.explainability.scoring_model import CorporateGenderScorer


def _ensure_synthetic_media_assets() -> dict:
    """Crea de forma determinista y reproducible archivos multimedia efímeros

    para verificar todos los desgloses granulares en Streamlit.
    """
    media_dir = Path("data/ephemeral_samples")
    media_dir.mkdir(parents=True, exist_ok=True)
    assets = {}

    # 1. Imágenes efímeras
    p_img1 = media_dir / "guardian_directora_1.jpg"
    if not p_img1.exists():
        im1 = Image.new("RGB", (280, 280), color=(240, 244, 248))
        d1 = ImageDraw.Draw(im1)
        d1.rectangle([30, 200, 250, 280], fill=(20, 50, 90))
        d1.ellipse([90, 60, 190, 175], fill=(225, 185, 160))
        d1.rectangle([110, 220, 170, 280], fill=(245, 245, 245))
        im1.save(p_img1, quality=90)
    assets["guardian_1"] = str(p_img1)

    p_img2 = media_dir / "guardian_directora_2.jpg"
    if not p_img2.exists():
        im2 = Image.new("RGB", (280, 280), color=(235, 240, 245))
        d2 = ImageDraw.Draw(im2)
        d2.rectangle([20, 180, 130, 280], fill=(30, 41, 59))
        d2.ellipse([45, 70, 105, 150], fill=(220, 180, 150))
        d2.rectangle([150, 180, 260, 280], fill=(14, 116, 144))
        d2.ellipse([175, 70, 235, 150], fill=(225, 185, 160))
        im2.save(p_img2, quality=90)
    assets["guardian_2"] = str(p_img2)

    p_img3 = media_dir / "wikimedia_consejo.jpg"
    if not p_img3.exists():
        im3 = Image.new("RGB", (280, 280), color=(245, 245, 240))
        d3 = ImageDraw.Draw(im3)
        d3.rectangle([50, 160, 230, 280], fill=(55, 65, 81))
        d3.ellipse([95, 50, 185, 155], fill=(230, 190, 165))
        im3.save(p_img3, quality=90)
    assets["wikimedia_1"] = str(p_img3)

    p_img4 = media_dir / "wikimedia_ingeniera_jefa.jpg"
    if not p_img4.exists():
        im4 = Image.new("RGB", (280, 280), color=(224, 242, 254))
        d4 = ImageDraw.Draw(im4)
        d4.rectangle([40, 170, 240, 280], fill=(15, 118, 110))
        d4.ellipse([90, 55, 190, 165], fill=(240, 200, 175))
        im4.save(p_img4, quality=90)
    assets["wikimedia_2"] = str(p_img4)

    # 2. Audios WAV reproducibles
    def _create_sine_wav(path: Path, freq: float, duration_sec: float = 1.5):
        if path.exists():
            return
        sample_rate = 16000
        num_samples = int(sample_rate * duration_sec)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            raw_data = bytearray()
            for s in range(num_samples):
                val = int(
                    12000.0 * math.sin(2.0 * math.pi * freq * (s / sample_rate))
                )
                raw_data.extend(struct.pack("<h", val))
            wf.writeframes(raw_data)

    p_aud1 = media_dir / "youtube_debate_bloque1_apertura.wav"
    _create_sine_wav(p_aud1, 440.0, 1.8)
    assets["yt_audio_1"] = str(p_aud1)

    p_aud2 = media_dir / "youtube_debate_bloque2_turno_analista.wav"
    _create_sine_wav(p_aud2, 554.37, 2.0)
    assets["yt_audio_2"] = str(p_aud2)

    p_aud3 = media_dir / "archive_cinta_historica_1998_declaracion.wav"
    _create_sine_wav(p_aud3, 330.0, 1.5)
    assets["archive_audio_1"] = str(p_aud3)

    p_aud4 = media_dir / "archive_cinta_historica_1998_rueda_prensa.wav"
    _create_sine_wav(p_aud4, 392.0, 1.5)
    assets["archive_audio_2"] = str(p_aud4)

    # 3. Clips de vídeo
    p_vid1 = media_dir / "youtube_debate_hd_camara_principal.mp4"
    if not p_vid1.exists():
        p_vid1.touch()
    assets["yt_video_1"] = str(p_vid1)

    p_vid2 = media_dir / "youtube_debate_hd_plano_analista.mp4"
    if not p_vid2.exists():
        p_vid2.touch()
    assets["yt_video_2"] = str(p_vid2)

    p_vid3 = media_dir / "archive_rollo_a_1998_intervencion.mp4"
    if not p_vid3.exists():
        p_vid3.touch()
    assets["archive_video_1"] = str(p_vid3)

    p_vid4 = media_dir / "archive_rollo_b_1998_conclusion.mp4"
    if not p_vid4.exists():
        p_vid4.touch()
    assets["archive_video_2"] = str(p_vid4)

    return assets


def _build_sample_doc(
    doc_id: str,
    title: str,
    url: str,
    source: MediaSource,
    company: str,
    subgroup: str,
    level: str,
    text_samples: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
) -> ExtractedFeaturesPayload:
    text_samples = text_samples or []
    image_paths = image_paths or []
    audio_paths = audio_paths or []
    video_paths = video_paths or []

    if level == "high":
        txt = TextMetrics(
            sentence_female_ratio=0.72,
            sentence_male_ratio=0.14,
            sentence_neutral_ratio=0.14,
            lexical_root_diversity=0.82,
            feminine_suffix_ratio=0.88,
            female_mention_frequency=18,
            relative_frequency_score=0.78,
            syntactic_agency_ratio=0.85,
            expert_quote_ratio=0.88,
            headline_presence=1.0,
        )
        aud = AudioMetrics(
            opened_by=GenderTarget.FEMALE,
            closed_by=GenderTarget.FEMALE,
            female_opened_score=1.0,
            female_closed_score=1.0,
            middle_segment_female_ratio=0.70,
            female_speaking_time_ratio=0.68,
            female_turn_count=14,
            speaking_turns_ratio=0.66,
            acoustic_gender_confidence=0.95,
            no_interruption_score=0.90,
        )
        img = ImageMetrics(
            visual_gender_female_ratio=0.75,
            shot_scale_importance_score=0.88,
            camera_angle_power_score=0.85,
            rule_of_thirds_alignment=0.90,
            shadow_lighting_prominence=0.86,
            cross_image_appearance_count=9,
            cross_image_persistence_ratio=0.82,
            spatial_centrality_score=0.86,
            leadership_visual_context=0.92,
        )
    elif level == "medium_high":
        txt = TextMetrics(
            sentence_female_ratio=0.62,
            sentence_male_ratio=0.20,
            sentence_neutral_ratio=0.18,
            lexical_root_diversity=0.74,
            feminine_suffix_ratio=0.76,
            female_mention_frequency=14,
            relative_frequency_score=0.68,
            syntactic_agency_ratio=0.72,
            expert_quote_ratio=0.74,
            headline_presence=1.0,
        )
        aud = AudioMetrics(
            opened_by=GenderTarget.FEMALE,
            closed_by=GenderTarget.NEUTRAL,
            female_opened_score=1.0,
            female_closed_score=0.5,
            middle_segment_female_ratio=0.60,
            female_speaking_time_ratio=0.58,
            female_turn_count=10,
            speaking_turns_ratio=0.56,
            acoustic_gender_confidence=0.92,
            no_interruption_score=0.80,
        )
        img = ImageMetrics(
            visual_gender_female_ratio=0.64,
            shot_scale_importance_score=0.75,
            camera_angle_power_score=0.72,
            rule_of_thirds_alignment=0.78,
            shadow_lighting_prominence=0.76,
            cross_image_appearance_count=6,
            cross_image_persistence_ratio=0.68,
            spatial_centrality_score=0.75,
            leadership_visual_context=0.80,
        )
    elif level == "medium":
        txt = TextMetrics(
            sentence_female_ratio=0.50,
            sentence_male_ratio=0.25,
            sentence_neutral_ratio=0.25,
            lexical_root_diversity=0.62,
            feminine_suffix_ratio=0.64,
            female_mention_frequency=9,
            relative_frequency_score=0.55,
            syntactic_agency_ratio=0.65,
            expert_quote_ratio=0.60,
            headline_presence=0.5,
        )
        aud = AudioMetrics(
            opened_by=GenderTarget.NEUTRAL,
            closed_by=GenderTarget.FEMALE,
            female_opened_score=0.5,
            female_closed_score=1.0,
            middle_segment_female_ratio=0.52,
            female_speaking_time_ratio=0.50,
            female_turn_count=8,
            speaking_turns_ratio=0.50,
            acoustic_gender_confidence=0.89,
            no_interruption_score=0.75,
        )
        img = ImageMetrics(
            visual_gender_female_ratio=0.55,
            shot_scale_importance_score=0.65,
            camera_angle_power_score=0.62,
            rule_of_thirds_alignment=0.68,
            shadow_lighting_prominence=0.70,
            cross_image_appearance_count=5,
            cross_image_persistence_ratio=0.58,
            spatial_centrality_score=0.65,
            leadership_visual_context=0.70,
        )
    elif level == "low_medium":
        txt = TextMetrics(
            sentence_female_ratio=0.28,
            sentence_male_ratio=0.52,
            sentence_neutral_ratio=0.20,
            lexical_root_diversity=0.35,
            feminine_suffix_ratio=0.32,
            female_mention_frequency=4,
            relative_frequency_score=0.30,
            syntactic_agency_ratio=0.30,
            expert_quote_ratio=0.25,
            headline_presence=0.0,
        )
        aud = AudioMetrics(
            opened_by=GenderTarget.MALE,
            closed_by=GenderTarget.NEUTRAL,
            female_opened_score=0.0,
            female_closed_score=0.5,
            middle_segment_female_ratio=0.30,
            female_speaking_time_ratio=0.28,
            female_turn_count=4,
            speaking_turns_ratio=0.26,
            acoustic_gender_confidence=0.86,
            no_interruption_score=0.45,
        )
        img = ImageMetrics(
            visual_gender_female_ratio=0.32,
            shot_scale_importance_score=0.35,
            camera_angle_power_score=0.38,
            rule_of_thirds_alignment=0.40,
            shadow_lighting_prominence=0.42,
            cross_image_appearance_count=2,
            cross_image_persistence_ratio=0.30,
            spatial_centrality_score=0.40,
            leadership_visual_context=0.30,
        )
    else:
        txt = TextMetrics(
            sentence_female_ratio=0.18,
            sentence_male_ratio=0.64,
            sentence_neutral_ratio=0.18,
            lexical_root_diversity=0.24,
            feminine_suffix_ratio=0.22,
            female_mention_frequency=2,
            relative_frequency_score=0.20,
            syntactic_agency_ratio=0.25,
            expert_quote_ratio=0.15,
            headline_presence=0.0,
        )
        aud = AudioMetrics(
            opened_by=GenderTarget.MALE,
            closed_by=GenderTarget.MALE,
            female_opened_score=0.0,
            female_closed_score=0.0,
            middle_segment_female_ratio=0.20,
            female_speaking_time_ratio=0.18,
            female_turn_count=2,
            speaking_turns_ratio=0.18,
            acoustic_gender_confidence=0.84,
            no_interruption_score=0.35,
        )
        img = ImageMetrics(
            visual_gender_female_ratio=0.22,
            shot_scale_importance_score=0.24,
            camera_angle_power_score=0.25,
            rule_of_thirds_alignment=0.28,
            shadow_lighting_prominence=0.30,
            cross_image_appearance_count=1,
            cross_image_persistence_ratio=0.20,
            spatial_centrality_score=0.30,
            leadership_visual_context=0.25,
        )

    vid = VideoMetrics(
        audio_track=aud,
        frame_set_visuals=img,
        optical_flow_motion_score=0.70 if level in ["high", "medium_high"] else 0.30,
        kinetic_dominance_ratio=0.68 if level in ["high", "medium_high"] else 0.28,
        silhouette_change_histogram_score=0.65 if level in ["high", "medium_high"] else 0.32,
        female_screen_time_ratio=0.72 if level in ["high", "medium_high"] else 0.30,
        active_speaker_sync_score=0.75 if level in ["high", "medium_high"] else 0.25,
    )

    if not (image_paths or audio_paths or video_paths):   # documento solo de texto
        final_txt, final_aud, final_img, final_vid = txt, None, None, None
    elif source == MediaSource.GUARDIAN:
        final_txt, final_aud, final_img, final_vid = txt, None, img, None
    elif source == MediaSource.WIKIMEDIA:
        final_txt, final_aud, final_img, final_vid = txt, None, img, None
    elif source == MediaSource.YOUTUBE:
        final_txt, final_aud, final_img, final_vid = None, aud, None, vid
    else:
        final_txt, final_aud, final_img, final_vid = txt, aud, img, vid

    # CONSTRUCCIÓN DE MUESTRAS ENRIQUECIDAS CON MÉTRICAS VARIADAS E INDEPENDIENTES
    enriched_texts = []
    for i, t in enumerate(text_samples):
        var_txt = TextMetrics(
            sentence_female_ratio=min(1.0, max(0.0, txt.sentence_female_ratio + (i * 0.05))),
            sentence_male_ratio=txt.sentence_male_ratio,
            sentence_neutral_ratio=txt.sentence_neutral_ratio,
            lexical_root_diversity=txt.lexical_root_diversity,
            feminine_suffix_ratio=txt.feminine_suffix_ratio,
            female_mention_frequency=max(0, txt.female_mention_frequency + i),
            relative_frequency_score=txt.relative_frequency_score,
            syntactic_agency_ratio=txt.syntactic_agency_ratio,
            expert_quote_ratio=txt.expert_quote_ratio,
            headline_presence=txt.headline_presence,
        ) if txt else None
        enriched_texts.append(TextSample(text=t, metrics=var_txt))

    enriched_images = []
    for i, p in enumerate(image_paths):
        var_img = ImageMetrics(
            visual_gender_female_ratio=min(1.0, max(0.0, img.visual_gender_female_ratio - (i * 0.08))),
            shot_scale_importance_score=img.shot_scale_importance_score,
            camera_angle_power_score=img.camera_angle_power_score,
            rule_of_thirds_alignment=img.rule_of_thirds_alignment,
            shadow_lighting_prominence=img.shadow_lighting_prominence,
            cross_image_appearance_count=img.cross_image_appearance_count,
            cross_image_persistence_ratio=img.cross_image_persistence_ratio,
            spatial_centrality_score=img.spatial_centrality_score,
            leadership_visual_context=img.leadership_visual_context,
        ) if img else None
        enriched_images.append(ImageSample(image_path=p, metrics=var_img))

    enriched_audios = []
    for i, a in enumerate(audio_paths):
        var_aud = AudioMetrics(
            opened_by=aud.opened_by,
            closed_by=aud.closed_by,
            female_opened_score=aud.female_opened_score,
            female_closed_score=aud.female_closed_score,
            middle_segment_female_ratio=min(1.0, max(0.0, aud.middle_segment_female_ratio + (i * 0.12))),
            female_speaking_time_ratio=min(1.0, max(0.0, aud.female_speaking_time_ratio - (i * 0.10))),
            female_turn_count=max(1, aud.female_turn_count + (i * 3)),
            speaking_turns_ratio=min(1.0, max(0.0, aud.speaking_turns_ratio + (i * 0.07))),
            acoustic_gender_confidence=aud.acoustic_gender_confidence,
            no_interruption_score=min(1.0, max(0.0, aud.no_interruption_score - (i * 0.05))),
        ) if aud else None
        enriched_audios.append(AudioSample(audio_path=a, metrics=var_aud))

    enriched_videos = []
    for i, v in enumerate(video_paths):
        var_vid = VideoMetrics(
            audio_track=vid.audio_track,
            frame_set_visuals=vid.frame_set_visuals,
            optical_flow_motion_score=min(1.0, max(0.0, vid.optical_flow_motion_score + (i * 0.15))),
            kinetic_dominance_ratio=min(1.0, max(0.0, vid.kinetic_dominance_ratio - (i * 0.08))),
            silhouette_change_histogram_score=vid.silhouette_change_histogram_score,
            female_screen_time_ratio=min(1.0, max(0.0, vid.female_screen_time_ratio + (i * 0.10))),
            active_speaker_sync_score=vid.active_speaker_sync_score,
        ) if vid else None
        enriched_videos.append(VideoSample(video_path=v, metrics=var_vid))

    primary_snippet = text_samples[0] if text_samples else ""

    return ExtractedFeaturesPayload(
        doc_id=doc_id,
        title=title,
        url=url,
        source=source,
        company_query=company,
        raw_text_snippet=primary_snippet,
        text_samples=enriched_texts,
        media_image_samples=enriched_images,
        media_audio_samples=enriched_audios,
        media_video_samples=enriched_videos,
        demographic_subgroup=subgroup,
        text_metrics=final_txt,
        audio_metrics=final_aud,
        image_metrics=final_img,
        video_metrics=final_vid,
        biometric_purged=True,
    )


def generate_mock_dataset(
    company: str = "Iberdrola",
) -> List[ExtractedFeaturesPayload]:
    assets = _ensure_synthetic_media_assets()

    samples = [
        _build_sample_doc(
            doc_id="doc_001",
            title=f"La directora ejecutiva de {company} lidera el plan de transición energética",
            url="https://theguardian.com/business/example1",
            source=MediaSource.GUARDIAN,
            company=company,
            subgroup="Mujer_Adulta_VozNativa",
            level="high",
            text_samples=[
                f"La directora ejecutiva y presidenta de la división en {company} explicó la estrategia de innovación y liderazgo femenino en el consejo.",
                f"Durante su comparecencia, la consejera delegada de {company} anunció un fondo pionero de I+D impulsado por ingenieras y científicas líderes.",
            ],
            image_paths=[assets["guardian_1"], assets["guardian_2"]],
            audio_paths=[],
            video_paths=[],
        ),
        _build_sample_doc(
            doc_id="doc_002",
            title=f"Resumen anual y debate financiero sobre {company} en televisión",
            url="https://youtube.com/watch?v=example2",
            source=MediaSource.YOUTUBE,
            company=company,
            subgroup="Mujer_Joven_MinoríaÉtnica",
            level="low",
            text_samples=[],
            image_paths=[],
            audio_paths=[assets["yt_audio_1"], assets["yt_audio_2"]],
            video_paths=[assets["yt_video_1"], assets["yt_video_2"]],
        ),
        _build_sample_doc(
            doc_id="doc_003",
            title=f"Historia corporativa y consejo de administración de {company}",
            url="https://es.wikipedia.org/wiki/Example3",
            source=MediaSource.WIKIMEDIA,
            company=company,
            subgroup="Mujer_Adulta_VozNativa",
            level="medium",
            text_samples=[
                f"La consejera independiente y científica fundadora anunció nuevos acuerdos tecnológicos estratégicos dentro de la junta directiva de {company}.",
                f"En la sección de gobierno corporativo se destaca a la directiva y jurista especializada como portavoz clave en la modernización operativa de {company}.",
            ],
            image_paths=[assets["wikimedia_1"], assets["wikimedia_2"]],
            audio_paths=[],
            video_paths=[],
        ),
        _build_sample_doc(
            doc_id="doc_004",
            title=f"Cobertura global de prensa sobre sostenibilidad en {company}",
            url="https://www.theguardian.com/example4",
            source=MediaSource.GUARDIAN,
            company=company,
            subgroup="Mujer_Senior_Directiva",
            level="medium_high",
            text_samples=[
                f"La jefa de operaciones y experta financiera de {company} lidera la cumbre internacional sobre descarbonización.",
                f"Analistas internacionales destacaron que la consejera ejecutiva de {company} estructuró el acuerdo de bonos verdes con respaldo unánime.",
                f"El informe corporativo sitúa a la directora de sostenibilidad de {company} como una referencia técnica en auditorías de impacto medioambiental.",
            ],
            image_paths=[],
            audio_paths=[],
            video_paths=[],
        ),
        _build_sample_doc(
            doc_id="doc_005",
            title=f"Hemeroteca audiovisual y archivo histórico de {company}",
            url="https://archive.org/details/example5",
            source=MediaSource.ARCHIVE,
            company=company,
            subgroup="Mujer_Joven_MinoríaÉtnica",
            level="low_medium",
            text_samples=[
                f"En el metraje de archivo institucional de 1998 de {company}, la técnica de laboratorio participa en calidad de apoyo en la mesa de trabajo.",
            ],
            image_paths=[assets["guardian_1"]],
            audio_paths=[assets["archive_audio_1"], assets["archive_audio_2"]],
            video_paths=[assets["archive_video_1"], assets["archive_video_2"]],
        ),
    ]

    out_path = Path("tests/mocks/mock_extracted_features.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            [s.model_dump(mode="json") for s in samples],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return samples


if __name__ == "__main__":
    docs = generate_mock_dataset("Iberdrola")
    scorer = CorporateGenderScorer()
    res = scorer.predict_importance_score(docs)
    auditor = IntersectionalFairnessAuditor()
    fairness = auditor.evaluate_fairness(res["dataframe"])
    print(
        f"[OK] Mock granular generado con éxito. Global Score: {res['global_score']}/100 |"
        f" Fairness: {fairness['status']}"
    )