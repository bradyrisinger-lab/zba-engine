from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
import csv

app = FastAPI(title="ZBA Engine", version="0.1.0")

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


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
    file_path = UPLOAD_DIR / request.file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="uploaded file not found")

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
