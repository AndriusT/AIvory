from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from google import genai
import anthropic


def init_gemini_client(api_key: Optional[str] = None):
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set.")
    return genai.Client(api_key=key)


def init_anthropic_client(api_key: Optional[str] = None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=key)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def gemini_classify_image(client, model_id: str, image_path: Path) -> Dict[str, Any]:
    image_to_analyze = Image.open(image_path).convert("RGB")

    buf = io.BytesIO()
    image_to_analyze.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    prompt = """
You are analyzing carved objects for forensic ivory identification.

Your task is NOT to guess based on object type.
Instead, analyze ONLY visible material evidence.

If evidence is insufficient, return uncertain.

Return ONLY valid JSON:
{
  "class": "ivory|bone|uncertain",
  "confidence": 0.0,
  "reasoning": "short explanation referencing visible evidence"
}
"""

    response = client.models.generate_content(
        model=model_id,
        contents=[
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_bytes,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
    )

    raw_text = response.text.strip()
    parsed = extract_json(raw_text)

    if parsed is None:
        return {
            "class": "uncertain",
            "confidence": None,
            "reasoning": "Could not parse Gemini JSON response.",
            "raw_text": raw_text,
        }

    return {
        "class": str(parsed.get("class", "uncertain")).strip().lower(),
        "confidence": parsed.get("confidence", None),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "raw_text": raw_text,
    }
