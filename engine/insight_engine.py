import json
import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def get_client():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY") or os.getenv("OPENAI_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def generate_insights(financial_context: dict) -> list[dict]:
    """Generate 10-30 structured financial insights from the provided context."""
    prompt = f"""You are an AI CFO assistant. Extract 10 to 30 structured financial insights from the data below.

Return ONLY valid JSON as an array of objects. Each object must use this schema:
{{
  "severity": "Low|Medium|High",
  "category": "string",
  "title": "string",
  "description": "string",
  "impact": "string",
  "estimated_savings": 0,
  "actionable": true,
  "resolved": false
}}

Data:
{json.dumps(financial_context, indent=2)}
"""

    try:
        client = get_client()
    except Exception:
        return []

    if client is None:
        return []

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always return valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception:
        return []

    return []
