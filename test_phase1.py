import os
import sys
from pathlib import Path
from sqlalchemy import inspect

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.database import init_db, engine, DB_PATH, REQUIRED_DIRS
from app.main import app
from fastapi.testclient import TestClient

def test_phase1():
    print("=== TESTING PHASE 1 ===")

    # 1. Test directory creation and database initialization
    print("\n1. Testing directory creation & SQLite init...")
    init_db()
    for d in REQUIRED_DIRS:
        assert d.exists(), f"Directory missing: {d}"
        print(f"  [OK] Directory exists: {d}")

    assert DB_PATH.exists(), f"Database file missing: {DB_PATH}"
    print(f"  [OK] Database file created at: {DB_PATH}")

    # 2. Check table schemas
    print("\n2. Inspecting SQLite tables...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"  Tables found: {tables}")
    expected_tables = ["products", "videos", "voices", "content", "publishing", "settings"]
    for t in expected_tables:
        assert t in tables, f"Expected table '{t}' not found in SQLite!"
        cols = [c["name"] for c in inspector.get_columns(t)]
        print(f"  [OK] Table '{t}' verified with columns: {cols}")

    # 3. Test FastAPI endpoints with TestClient
    print("\n3. Testing FastAPI endpoints...")
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    print("  [OK] /health responded 200:", res.json())

    # Dashboard
    res = client.get("/dashboard")
    assert res.status_code == 200, f"Dashboard failed: {res.status_code}"
    assert "AI CONTENT FACTORY" in res.text, "Brand text missing from dashboard HTML"
    assert "Dashboard" in res.text, "Dashboard title missing from HTML"
    print("  [OK] /dashboard responded 200 and rendered HTML properly")

    # Root route
    res = client.get("/")
    assert res.status_code == 200, f"Root / failed: {res.status_code}"
    print("  [OK] / responded 200")

    # Other pages
    routes_to_test = [
        "/research",
        "/products",
        "/videos",
        "/review",
        "/voice",
        "/content",
        "/publishing",
        "/settings"
    ]
    for r in routes_to_test:
        res = client.get(r)
        assert res.status_code == 200, f"Route {r} failed with status {res.status_code}"
        print(f"  [OK] Route {r} responded 200")

    # 4. Check START.bat
    print("\n4. Checking START.bat...")
    start_bat = Path(__file__).parent / "START.bat"
    assert start_bat.exists(), "START.bat does not exist!"
    content = start_bat.read_text(encoding="utf-8")
    assert "uvicorn" in content, "START.bat missing uvicorn call"
    assert "http://localhost:8000" in content, "START.bat missing localhost url"
    print("  [OK] START.bat exists and contains valid launch commands")

    print("\n=== ALL PHASE 1 TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_phase1()
