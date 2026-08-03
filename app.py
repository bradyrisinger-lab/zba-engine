from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import csv
import io

app = FastAPI(title="ZBA Engine", version="0.1.0")


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
