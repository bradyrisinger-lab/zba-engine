import os
from pathlib import Path

from fastapi.testclient import TestClient

from app import UPLOAD_DIR, app


os.environ.setdefault("OPENAI_API_KEY", "dummy-key")
client = TestClient(app)


def test_analyze_returns_summary_for_uploaded_csv(tmp_path):
    UPLOAD_DIR.mkdir(exist_ok=True)
    sample_path = UPLOAD_DIR / "sample_test.csv"
    sample_path.write_text(
        "Amount,Category,Date\n"
        "100000,Engineering,2024-01-01\n"
        "50000,Marketing,2024-01-02\n"
        "20000,Operations,2024-01-03\n",
        encoding="utf-8",
    )

    response = client.post(
        "/analyze",
        json={
            "file_name": "sample_test.csv",
            "revenue_column": "Amount",
            "category_column": "Category",
            "date_column": "Date",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["revenue"] == 170000.0
    assert data["largest_category"] == "Engineering"
    assert data["by_category"]["Engineering"] == 100000.0

    sample_path.unlink(missing_ok=True)


def test_ai_report_returns_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import app as app_module

    app_module.client = None

    response = client.post(
        "/ai-report",
        json={
            "file_name": "sample.csv",
            "revenue_column": "Amount",
            "category_column": "Category",
            "date_column": "Date",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
