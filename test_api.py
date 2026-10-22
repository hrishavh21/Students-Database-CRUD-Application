"""Run with:  pytest -q"""
import os, tempfile
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["SEED"] = "0"

import httpx
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
S = {"name": "Test User", "date_of_birth": "2004-05-01", "email": "t@example.com", "course": "Python"}


def test_crud_flow():
    r = client.post("/students", json=S)                       # Create
    assert r.status_code == 201
    sid = r.json()["id"]
    assert client.get(f"/students/{sid}").json()["name"] == "Test User"          # Read one
    assert len(client.get("/students").json()) == 1                              # Read all
    r = client.patch(f"/students/{sid}", json={"course": "AI"})                  # Update (partial)
    assert r.json()["course"] == "AI" and r.json()["email"] == "t@example.com"
    r = client.put(f"/students/{sid}", json={**S, "name": "New Name"})           # Update (full)
    assert r.json()["name"] == "New Name"
    assert client.delete(f"/students/{sid}").status_code == 204                  # Delete
    assert client.get(f"/students/{sid}").status_code == 404


def test_validation_and_errors():
    assert client.post("/students", json={**S, "email": "bad"}).status_code == 422          # bad email
    assert client.post("/students", json={"name": "A", "date_of_birth": "2004-01-01"}).status_code == 422  # no contact
    assert client.post("/students", json={**S, "date_of_birth": "2999-01-01"}).status_code == 422  # future DOB
    new_id = client.post("/students", json={**S, "email": "dup@example.com"}).json()["id"]
    assert client.post("/students", json={**S, "email": "dup@example.com"}).status_code == 409     # duplicate
    assert client.patch("/students/9999", json={"course": "X"}).status_code == 404
    assert client.delete("/students/9999").status_code == 404
    assert client.patch(f"/students/{new_id}", json={"email": "bad"}).status_code == 422           # patch is validated too


def test_filters():
    client.post("/students", json={**S, "name": "Zed", "email": "z@example.com", "date_of_birth": "2008-01-01", "course": "Web"})
    assert [s["name"] for s in client.get("/students", params={"born_after": "2007-12-31"}).json()] == ["Zed"]
    assert len(client.get("/students", params={"course": "web"}).json()) == 1
    assert len(client.get("/students", params={"name": "ze"}).json()) == 1


def test_sql_injection_is_harmless():
    assert client.get("/students", params={"name": "'; DROP TABLE students; --"}).json() == []
    assert client.get("/students").status_code == 200


def test_ask_is_grounded_and_handles_ollama_down(monkeypatch):
    seen = {}
    def fake_post(url, json, timeout):
        seen["prompt"] = json["prompt"]
        class R:
            def raise_for_status(self): pass
            def json(self): return {"response": "There is 1 student."}
        return R()
    monkeypatch.setattr(main.httpx, "post", fake_post)
    r = client.post("/ask", json={"question": "List students born after 2007"})
    assert r.status_code == 200 and r.json()["matching_students"] == 1 and "Zed" in seen["prompt"]

    def down(*a, **k): raise httpx.ConnectError("down")
    monkeypatch.setattr(main.httpx, "post", down)
    assert client.post("/ask", json={"question": "How many students are there?"}).status_code == 503
