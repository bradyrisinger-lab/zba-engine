from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
import csv

app = FastAPI(title="ZBA Engine", version="0.1.0")

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class CalculationRequest(BaseModel):
    values: list[float]
    mode: str = "sum"


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
