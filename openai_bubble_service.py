import base64
import json
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def image_url_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_bubble_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "image_summary": {"type": "string"},
            "scale_bar_px": {
                "anyOf": [{"type": "number"}, {"type": "null"}]
            },
            "bubbles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "radius_px": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["x", "y", "radius_px", "confidence"],
                },
            },
        },
        "required": ["image_summary", "scale_bar_px", "bubbles"],
    }


def analyze_bubbles_with_openai(
    api_key: str,
    image_bytes: bytes,
    mime_type: str = "image/png",
    model: str = "gpt-5.4",
) -> Dict[str, Any]:
    """
    Envia uma imagem para a OpenAI Responses API e pede JSON estruturado
    com centros e raios das bolhas.
    """
    client = OpenAI(api_key=api_key)
    data_url = image_url_to_data_url(image_bytes, mime_type=mime_type)

    prompt = """
Você está analisando uma imagem microscópica de microbolhas/espuma.

Tarefa:
1. Considere apenas a área principal útil da imagem.
2. Ignore texto superior, bordas irrelevantes e a barra de escala como bolha.
3. Identifique bolhas visíveis e retorne:
   - x: coordenada horizontal do centro em pixels
   - y: coordenada vertical do centro em pixels
   - radius_px: raio aproximado em pixels
   - confidence: confiança entre 0 e 1
4. Se conseguir identificar a barra de 1.0 mm, retorne o comprimento em pixels em scale_bar_px.
5. Retorne SOMENTE o JSON solicitado.

Regras:
- Prefira círculos que correspondam ao contorno visual dominante de cada bolha.
- Não invente bolhas fora da área útil.
- Não inclua a barra de escala nem o cabeçalho superior.
- confidence deve ser mais alta para bolhas visualmente claras.
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "bubble_detection",
                "schema": build_bubble_schema(),
                "strict": True,
            }
        },
    )

    raw_text = getattr(response, "output_text", None)
    if not raw_text:
        raise ValueError("A resposta da API veio sem output_text.")

    data = json.loads(raw_text)

    if "bubbles" not in data:
        raise ValueError("JSON retornado sem a chave 'bubbles'.")

    return data


def filter_bubbles(
    bubbles: List[Dict[str, Any]],
    min_radius_px: float = 3.0,
    min_confidence: float = 0.35,
) -> List[Dict[str, Any]]:
    filtered = []
    for b in bubbles:
        try:
            x = float(b["x"])
            y = float(b["y"])
            r = float(b["radius_px"])
            c = float(b["confidence"])
        except Exception:
            continue

        if r >= min_radius_px and c >= min_confidence:
            filtered.append(
                {
                    "x": x,
                    "y": y,
                    "radius_px": r,
                    "confidence": c,
                }
            )
    return filtered


def bubbles_to_rows(
    bubbles: List[Dict[str, Any]],
    px_per_mm: Optional[float],
) -> List[Dict[str, Any]]:
    rows = []
    for idx, b in enumerate(bubbles, start=1):
        radius_px = float(b["radius_px"])
        diameter_px = radius_px * 2.0

        if px_per_mm and px_per_mm > 0:
            diameter_um = (diameter_px / px_per_mm) * 1000.0
        else:
            diameter_um = None

        rows.append(
            {
                "Bolha": idx,
                "Centro X (px)": round(float(b["x"]), 1),
                "Centro Y (px)": round(float(b["y"]), 1),
                "Raio (px)": round(radius_px, 2),
                "Diâmetro (px)": round(diameter_px, 2),
                "Diâmetro (µm)": None if diameter_um is None else round(diameter_um, 2),
                "Confiança": round(float(b["confidence"]), 3),
            }
        )
    return rows
