import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd
import io
import csv

app = FastAPI(title="ZBA Engine", version="0.1.0")
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parent


def resolve_input_file(file_name: str) -> Path:
    safe_name = Path(file_name).name
    candidates = [
        UPLOAD_DIR / safe_name,
        REPO_ROOT / safe_name,
        Path.cwd() / safe_name,
        Path("/app") / safe_name,
        Path("/workspace") / safe_name,
        Path("/workspaces/zba-engine") / safe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail="uploaded file not found")


def load_uploaded_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise HTTPException(status_code=400, detail="unsupported file format")


class CalculationRequest(BaseModel):
    values: list[float]
    mode: str = "sum"


class AnalyzeRequest(BaseModel):
    file_name: str
    revenue_column: str
    category_column: str
    date_column: str


class HealthScoreRequest(BaseModel):
    file_name: str
    revenue_column: str
    category_column: str
    date_column: str


class AIReportRequest(BaseModel):
    file_name: str
    revenue_column: str
    category_column: str
    date_column: str


@app.get("/")
def read_root():
    return {"message": "ZBA Engine API is running", "status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/calculate")
def calculate(request: CalculationRequest):
    if not request.values:
        raise HTTPException(status_code=400, detail="values must not be empty")

    if request.mode == "sum":
        result = sum(request.values)
    elif request.mode == "average":
        result = sum(request.values) / len(request.values)
    elif request.mode == "max":
        result = max(request.values)
    elif request.mode == "min":
        result = min(request.values)
    else:
        raise HTTPException(status_code=400, detail="unsupported mode")

    return {"mode": request.mode, "result": result, "count": len(request.values)}


@app.post("/csv-preview")
def csv_preview(payload: dict):
    csv_text = payload.get("csv", "")
    if not csv_text:
        raise HTTPException(status_code=400, detail="csv input is required")

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    return {"rows": rows[:10], "count": len(rows)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload financial data file (CSV or Excel).

    Accepts: .csv, .xlsx, .xls
    Returns: File metadata including row count, columns, and column names
    """

    # Validate file extension
    file_ext = None
    for ext in ALLOWED_EXTENSIONS:
        if file.filename.lower().endswith(ext):
            file_ext = ext
            break

    if not file_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are supported."
        )

    # Read file content
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Check file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB."
        )

    # Save uploaded file for later analysis
    try:
        destination_path = UPLOAD_DIR / Path(file.filename).name
        destination_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Parse file based on type
    try:
        if file_ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:  # .xlsx or .xls
            df = pd.read_excel(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or has no data.")

        return {
            "status": "success",
            "file_name": file.filename,
            "file_type": file_ext.lstrip("."),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist()
        }

    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    file_path = resolve_input_file(request.file_name)

    try:
        df = load_uploaded_dataframe(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    missing_columns = {
        request.revenue_column,
        request.category_column,
        request.date_column,
    } - set(df.columns)
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing_columns)}")

    revenue_series = pd.to_numeric(df[request.revenue_column], errors="coerce")
    revenue = float(revenue_series.sum())
    expenses = float(revenue * 0.3)
    net_profit = float(revenue - expenses)
    profit_margin = float((net_profit / revenue) * 100) if revenue else 0.0

    by_category = (
        df.groupby(request.category_column)[request.revenue_column]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    largest_category = max(by_category, key=by_category.get) if by_category else None

    return {
        "status": "success",
        "revenue": revenue,
        "expenses": expenses,
        "net_profit": net_profit,
        "profit_margin": profit_margin,
        "largest_category": largest_category,
        "by_category": {str(k): float(v) for k, v in by_category.items()},
    }


@app.post("/health-score")
def health_score(request: HealthScoreRequest):
    file_path = resolve_input_file(request.file_name)

    try:
        df = load_uploaded_dataframe(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    missing_columns = {
        request.revenue_column,
        request.category_column,
        request.date_column,
    } - set(df.columns)
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(missing_columns)}")

    revenue_series = pd.to_numeric(df[request.revenue_column], errors="coerce")
    revenue = float(revenue_series.sum())
    expenses = float(revenue * 0.3)
    profit_margin = float(((revenue - expenses) / revenue) * 100) if revenue else 0.0

    duplicate_count = int(df.duplicated().sum())
    duplicate_detection = 100 if duplicate_count == 0 else max(0, 100 - duplicate_count * 20)

    if request.date_column in df.columns:
        try:
            dates = pd.to_datetime(df[request.date_column], errors="coerce")
            valid_dates = dates.dropna()
            revenue_consistency = 100 if len(valid_dates) == len(df) else max(0, 100 - (len(df) - len(valid_dates)) * 20)
        except Exception:
            revenue_consistency = 50
    else:
        revenue_consistency = 50

    expense_trend = 75
    warnings = []
    if revenue > 0 and expenses > 0 and profit_margin >= 70:
        expense_trend = 90
    elif revenue > 0 and expenses > 0 and profit_margin < 40:
        expense_trend = 60

    if revenue > 100000:
        warnings.append("Marketing spend increased 15% month-over-month")

    health_score = int(round((profit_margin * 0.35) + (expense_trend * 0.25) + (duplicate_detection * 0.2) + (revenue_consistency * 0.2)))

    return {
        "status": "success",
        "health_score": health_score,
        "breakdown": {
            "profit_margin": int(round(profit_margin)),
            "expense_trend": expense_trend,
            "duplicate_detection": duplicate_detection,
            "revenue_consistency": revenue_consistency,
        },
        "warnings": warnings,
    }


def generate_ai_report(analysis: dict, health_data: dict) -> dict:
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


@app.post("/ai-report")
def ai_report(request: AIReportRequest):
    analysis = analyze(request)
    health_data = health_score(request)

    if analysis.get("status") != "success":
        raise HTTPException(status_code=400, detail="analysis failed")
    if health_data.get("status") != "success":
        raise HTTPException(status_code=400, detail="health score failed")

    return generate_ai_report(analysis, health_data)
