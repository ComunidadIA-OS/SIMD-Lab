import html
from pathlib import Path
import re
from typing import Union
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

POSITIVE_LEADERSHIP_WORDS = {
    "directora",
    "presidenta",
    "ceo",
    "experta",
    "lidera",
    "anunció",
    "explicó",
    "ejecutiva",
    "científica",
    "fundadora",
    "consejera",
    "liderazgo",
    "innovación",
    "jefa",
}
PASSIVE_SECONDARY_WORDS = {
    "acompañada",
    "asistente",
    "apoyo",
    "secundaria",
    "esposa",
    "invitada",
    "mencionada",
}


class LimeMultimodalEngine:

  def __init__(self):
    self.text_explainer = LimeTextExplainer(
        class_names=[
            "Rol Secundario / Baja Relevancia",
            "Rol Protagonista / Alta Importancia",
        ]
    )
    self.image_explainer = LimeImageExplainer()

  # =========================================================================
  # 1. EXPLICABILIDAD DE TEXTO (LIME)
  # =========================================================================
  def _surrogate_text_classifier(self, texts):
    probs = []
    for t in texts:
      words = [w.strip('.,;:()"\'').lower() for w in t.split()]
      pos_hits = sum(1 for w in words if w in POSITIVE_LEADERSHIP_WORDS)
      neg_hits = sum(1 for w in words if w in PASSIVE_SECONDARY_WORDS)
      score_pos = 1 / (1 + np.exp(-(pos_hits * 0.9 - neg_hits * 0.8 - 0.2)))
      probs.append([1.0 - score_pos, score_pos])
    return np.array(probs)

  def explain_text_instance(self, text: str, num_features: int = 8) -> str:
    exp = self.text_explainer.explain_instance(
        text, self._surrogate_text_classifier, num_features=num_features
    )

    features = exp.as_list(label=1)
    weights = {word.lower(): weight for word, weight in features}
    max_weight = (
        max((abs(weight) for _, weight in features), default=1.0) or 1.0
    )

    probabilities = exp.predict_proba
    class_names = self.text_explainer.class_names

    def probability_row(name, probability, color):
      pct = max(0.0, min(1.0, float(probability))) * 100
      return f"""
            <div class="prob-row">
                <div class="prob-label">
                    <span>{html.escape(name)}</span>
                    <strong>{pct:.1f}%</strong>
                </div>
                <div class="track">
                    <div class="fill" style="width:{pct:.2f}%;background:{color}"></div>
                </div>
            </div>
            """

    probabilities_html = probability_row(
        class_names[0], probabilities[0], "#64748b"
    ) + probability_row(class_names[1], probabilities[1], "#0ea5e9")

    feature_rows = []
    for word, weight in sorted(
        features, key=lambda item: abs(item[1]), reverse=True
    ):
      width = abs(weight) / max_weight * 100
      positive = weight >= 0
      color = "#0ea5e9" if positive else "#f43f5e"
      sign = "+" if positive else ""

      feature_rows.append(f"""
            <div class="feature-row">
                <span class="feature-name">{html.escape(word)}</span>
                <div class="feature-track">
                    <div class="feature-fill" style="width:{width:.2f}%;background:{color}"></div>
                </div>
                <span class="weight">{sign}{weight:.3f}</span>
            </div>
            """)

    def highlight_token(match):
      token = match.group(0)
      weight = weights.get(token.lower())
      if weight is None:
        return html.escape(token)

      color = "#0ea5e9" if weight >= 0 else "#f43f5e"
      background = "#e0f2fe" if weight >= 0 else "#ffe4e6"

      return (
          f'<mark title="Impacto: {weight:+.3f}"'
          f' style="background:{background};color:#0f172a;border-bottom:2.5px'
          f' solid'
          f' {color};padding:1px 3px;border-radius:4px;font-weight:600;">{html.escape(token)}</mark>'
      )

    highlighted_text = re.sub(
        r"\w+|[^\w\s]+|\s+", highlight_token, text, flags=re.UNICODE
    )

    return f"""
        <div class="lime-container" style="padding: 16px; font-family: 'Montserrat', sans-serif; color: #1e293b; background: #ffffff; font-size: 13px; border: 1px solid #e2e8f0; border-radius: 10px;">
            <style>
                .lime-container * {{ box-sizing: border-box; }}
                .lime-container .section {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 12px; background: #fafafa; }}
                .lime-container h3 {{ font-size: 14px; margin: 0 0 10px; color: #0f172A; font-weight: 700; }}
                .lime-container .prob-row {{ margin: 8px 0; }}
                .lime-container .prob-label {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }}
                .lime-container .track, .lime-container .feature-track {{ background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
                .lime-container .track {{ height: 8px; }}
                .lime-container .fill {{ height: 100%; border-radius: 4px; }}
                .lime-container .feature-row {{ display: grid; grid-template-columns: minmax(80px, 1.2fr) minmax(60px, 2fr) 60px; gap: 12px; align-items: center; margin: 8px 0; }}
                .lime-container .feature-name {{ overflow-wrap: anywhere; font-weight: 600; color: #1e293b; font-size: 12px; }}
                .lime-container .feature-track {{ height: 10px; }}
                .lime-container .feature-fill {{ height: 100%; border-radius: 4px; }}
                .lime-container .weight {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-size: 12px; }}
                .lime-container .legend {{ color: #64748b; font-size: 11px; margin-top: 10px; }}
                .lime-container .explained-text {{ font-family: 'Merriweather', serif; font-size: 14px; line-height: 1.8; color: #1e293b; background: #ffffff; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0; }}
            </style>
            <div class="section">
                <h3>Probabilidad estimada por el modelo de NLP</h3>
                {probabilities_html}
            </div>
            <div class="section">
                <h3>Aporte de raíces léxicas y palabras activas</h3>
                {''.join(feature_rows)}
                <div class="legend">
                    <span style="color:#0ea5e9;font-weight:bold;">■ Azul</span>: Incrementa la relevancia. <span style="color:#f43f5e;font-weight:bold;">■ Rojo</span>: Reduce la relevancia.
                </div>
            </div>
            <div class="section" style="margin-bottom: 0;">
                <h3>Texto auditado con resaltado interactivo</h3>
                <div class="explained-text">{highlighted_text}</div>
            </div>
        </div>
        """

  # =========================================================================
  # 2. EXPLICABILIDAD DE IMAGEN (SIN OPENCV / EFÍMERO Y NATIVO)
  # =========================================================================
  def _load_image_rgb(
      self, image_input: Union[str, Path, Image.Image, np.ndarray]
  ) -> np.ndarray:
    """Carga y reescala a (224, 224) utilizando exclusivamente Pillow y NumPy."""
    if isinstance(image_input, (str, Path)):
      p = Path(image_input)
      if not p.exists():
        raise FileNotFoundError(f"Archivo de imagen no encontrado: {p}")
      pil_img = Image.open(p).convert("RGB")
    elif isinstance(image_input, Image.Image):
      pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
      if image_input.ndim == 2:
        pil_img = Image.fromarray(image_input).convert("RGB")
      elif image_input.shape[2] == 4:
        pil_img = Image.fromarray(image_input).convert("RGB")
      else:
        pil_img = Image.fromarray(image_input)
    else:
      raise ValueError(f"Tipo no soportado: {type(image_input)}")

    pil_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    return np.array(pil_resized)

  def explain_image_instance(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        num_features: int = 4,
        num_samples: int = 80,
    ) -> np.ndarray:
        """Ejecuta LIME sobre una imagen real explicando la densidad de rasgos en el encuadre."""
        img_rgb = self._load_image_rgb(image_input)

        def _vision_predict(images):
            preds = []
            h, w = images[0].shape[:2]
            y1, y2 = int(h * 0.2), int(h * 0.8)
            x1, x2 = int(w * 0.2), int(w * 0.8)

            for im in images:
                foco = im[y1:y2, x1:x2]
                # Densidad de rasgos activos en el encuadre principal
                actividad = float(np.mean(foco > 30))
                score = np.clip(actividad, 0.05, 0.95)
                preds.append([1.0 - score, score])
            return np.array(preds)

        explanation = self.image_explainer.explain_instance(
            img_rgb,
            _vision_predict,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
        )