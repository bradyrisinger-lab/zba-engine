import json
import os

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - defensive for environments without the package
    OpenAI = None


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def generate_ai_report(analysis: dict, health_data: dict) -> dict:
    """
    Generate an AI CFO report based on financial analysis and health score.

    Args:
        analysis: dict from /analyze endpoint
        health_data: dict from /health-score endpoint

    Returns:
        dict with executive_summary, strengths, risks, recommendations, confidence
    """
    financial_context = {
        "revenue": analysis.get("revenue"),
        "expenses": analysis.get("expenses"),
        "net_profit": analysis.get("net_profit"),
        "profit_margin": analysis.get("profit_margin"),
        "largest_category": analysis.get("largest_category"),
        "by_category": analysis.get("by_category"),
        "health_score": health_data.get("health_score"),
        "health_breakdown": health_data.get("breakdown"),
        "warnings": health_data.get("warnings"),
    }

    prompt = f"""You are the AI CFO for a small business.

Analyze ONLY the financial data provided below.

Never invent numbers, transactions, percentages, or trends.

If information is missing, explicitly state that it is unavailable.

Produce:
1. Executive Summary (1-2 sentences)
2. Strengths (list 2-3 key strengths based on the data)
3. Risks (list 2-3 key risks or concerns)
4. Recommendations (list 2-3 actionable recommendations)
5. Confidence Assessment (High/Medium/Low based on data completeness)

Financial Data:
{json.dumps(financial_context, indent=2)}

Respond ONLY with valid JSON in this exact format:
{{
  "executive_summary": "...",
  "strengths": ["...", "..."],
  "risks": ["...", "..."],
  "recommendations": ["...", "..."],
  "confidence": "High|Medium|Low"
}}"""

    client = get_client()
    if client is None:
        return {"status": "error", "message": "OpenAI API key is not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        ai_text = response.choices[0].message.content.strip()
        ai_report = json.loads(ai_text)
        return {"status": "success", "ai_report": ai_report}
    except json.JSONDecodeError:
        return {"status": "error", "message": "Failed to parse AI response as JSON", "raw_response": ai_text if 'ai_text' in locals() else None}
    except Exception as e:
        return {"status": "error", "message": str(e)}
