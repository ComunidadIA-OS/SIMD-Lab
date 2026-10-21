from typing import List
from src.common.schemas import ExtractedFeaturesPayload
from tests.test_xai_fairness_mock import generate_mock_dataset

USE_MOCK_DATA = True

def fetch_analyzed_documents(company: str, top_n: int, selected_sources: List[str]) -> List[ExtractedFeaturesPayload]:
    """Devuelve los documentos analizados filtrados por fuente y límite Top-N."""
    all_docs = generate_mock_dataset(company)
    filtered = [d for d in all_docs if d.source.value in selected_sources]
    return (filtered or all_docs)[:top_n]