from typing import List

from src.analyzers.multimodal_pipeline import load_payloads
from src.common.schemas import ExtractedFeaturesPayload
from tests.test_xai_fairness_mock import generate_mock_dataset

# True: datos inventados del generador mock. False: JSON que genera el pipeline de análisis.
USE_MOCK_DATA = False
#ANALYZED_DATA_PATH = "tests/mocks/real_extracted_features.json" #Datos reales del pipeline de análisis (DESCOMENTAR CUANDO ESTE LA PRIMERA PARTE)
ANALYZED_DATA_PATH = "tests/mocks/mock_pipeline_features.json" #Datos simulados del pipeline de análisis


def fetch_analyzed_documents(company: str, top_n: int, selected_sources: List[str]) -> List[ExtractedFeaturesPayload]:
    """Devuelve los documentos analizados filtrados por fuente y límite Top-N."""
    if USE_MOCK_DATA:
        all_docs = generate_mock_dataset(company)
    else:
        all_docs = load_payloads(ANALYZED_DATA_PATH)
    filtered = [d for d in all_docs if d.source.value in selected_sources]
    return (filtered or all_docs)[:top_n]
