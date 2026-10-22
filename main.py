"""Students Database CRUD API (FastAPI + SQLite) with an optional local-AI (Ollama) endpoint."""
import os
import re
import sqlite3
from contextlib import closing
from datetime import date
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

DB = os.getenv("DB_PATH", "students.db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

app = FastAPI(title="Students CRUD API", version="1.0")

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


# ---------------------------------------------------------------- database
def run(sql: str, args=()):
    """Run one SQL statement with bound parameters. Returns (rows, last_id, rows_changed)."""
    with closing(sqlite3.connect(DB)) as con:
        con.row_factory = sqlite3.Row
        with con:  # commits on success, rolls back on error
            cur = con.execute(sql, args)
            return [dict(r) for r in cur.fetchall()], cur.lastrowid, cur.rowcount


run("""CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date_of_birth TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        course TEXT,
        address TEXT,
        enrollment_date TEXT)""")


# ---------------------------------------------------------------- schemas
class StudentIn(BaseModel):
    model_config = {"json_schema_extra": {"example": {  # pre-fills a VALID body in the /docs page
        "name": "Anita Rao", "date_of_birth": "2005-04-10", "email": "anita@example.com",
        "phone": "9876543210", "course": "Python", "address": "Mumbai", "enrollment_date": "2024-07-01"}}}
    name: str = Field(min_length=1, max_length=100)
    date_of_birth: date
    email: Optional[str] = Field(None, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    phone: Optional[str] = Field(None, pattern=r"^\+?[0-9 \-]{7,15}$")
    course: Optional[str] = Field(None, max_length=100)
    address: Optional[str] = Field(None, max_length=200)
    enrollment_date: Optional[date] = None

    @field_validator("date_of_birth")
    @classmethod
    def dob_in_past(cls, v):
        if v >= date.today():
            raise ValueError("date_of_birth must be in the past")
        return v

    @model_validator(mode="after")
    def need_contact(self):
        if not (self.email or self.phone):
            raise ValueError("provide at least one of email or phone")
        return self


class Student(StudentIn):
    id: int


class StudentPatch(BaseModel):  # every field optional; merged and re-validated in the handler
    name: Optional[str] = None
    date_of_birth: Optional[date] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    course: Optional[str] = None
    address: Optional[str] = None
    enrollment_date: Optional[date] = None


FIELDS = list(StudentIn.model_fields)


def get_or_404(student_id: int) -> dict:
    rows, _, _ = run("SELECT * FROM students WHERE id = ?", (student_id,))
    if not rows:
        raise HTTPException(404, f"Student {student_id} not found")
    return rows[0]


def save(student_id: int, s: StudentIn) -> dict:
    sets = ", ".join(f"{f} = :{f}" for f in FIELDS)
    try:
        run(f"UPDATE students SET {sets} WHERE id = :id", {**s.model_dump(mode="json"), "id": student_id})
    except sqlite3.IntegrityError:
        raise HTTPException(409, "A student with this email already exists")
    return get_or_404(student_id)


# ---------------------------------------------------------------- CRUD endpoints
@app.post("/students", response_model=Student, status_code=201)
def create_student(s: StudentIn):
    cols, params = ", ".join(FIELDS), ", ".join(f":{f}" for f in FIELDS)
    try:
        _, new_id, _ = run(f"INSERT INTO students ({cols}) VALUES ({params})", s.model_dump(mode="json"))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "A student with this email already exists")
    return get_or_404(new_id)


@app.get("/students", response_model=list[Student])
def list_students(
    name: Optional[str] = Query(None, description="partial, case-insensitive match"),
    course: Optional[str] = Query(None, description="exact course, case-insensitive"),
    born_after: Optional[date] = None,
    born_before: Optional[date] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    where, args = [], []
    if name:
        where.append("name LIKE ?"); args.append(f"%{name}%")
    if course:
        where.append("course = ? COLLATE NOCASE"); args.append(course)
    if born_after:
        where.append("date_of_birth > ?"); args.append(born_after.isoformat())
    if born_before:
        where.append("date_of_birth < ?"); args.append(born_before.isoformat())
    sql = "SELECT * FROM students" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id LIMIT ? OFFSET ?"
    return run(sql, (*args, limit, offset))[0]


@app.get("/students/{student_id}", response_model=Student)
def read_student(student_id: int):
    return get_or_404(student_id)


@app.put("/students/{student_id}", response_model=Student)
def replace_student(student_id: int, s: StudentIn):  # full update
    get_or_404(student_id)
    return save(student_id, s)


@app.patch("/students/{student_id}", response_model=Student)
def update_student(student_id: int, patch: StudentPatch):  # partial update
    merged = {**get_or_404(student_id), **patch.model_dump(exclude_unset=True)}
    try:
        s = StudentIn(**{k: merged[k] for k in FIELDS})
    except ValidationError as e:
        raise HTTPException(422, [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()])
    return save(student_id, s)


@app.delete("/students/{student_id}", status_code=204)
def delete_student(student_id: int):
    get_or_404(student_id)
    run("DELETE FROM students WHERE id = ?", (student_id,))


# ---------------------------------------------------------------- optional bonus: local AI (Ollama)
class Question(BaseModel):
    question: str = Field(min_length=3, max_length=300)


@app.post("/ask")
def ask(q: Question):
    """The backend (not the model) fetches records with fixed, parameterised SQL; the model only
    phrases an answer from them. The model never writes or runs SQL."""
    text = q.question.lower()
    where, args = [], []
    if m := re.search(r"born (after|before|in) (\d{4})", text):
        word, year = m.group(1), m.group(2)
        if word == "after":
            where.append("date_of_birth > ?"); args.append(f"{year}-12-31")
        elif word == "before":
            where.append("date_of_birth < ?"); args.append(f"{year}-01-01")
        else:
            where.append("date_of_birth BETWEEN ? AND ?"); args += [f"{year}-01-01", f"{year}-12-31"]
    for r in run("SELECT DISTINCT course FROM students WHERE course IS NOT NULL")[0]:
        if r["course"].lower() in text:
            where.append("course = ? COLLATE NOCASE"); args.append(r["course"])
    sql = "SELECT * FROM students" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id"
    rows = run(sql, args)[0]
    shown = rows[:50]
    context = f"Number of matching students: {len(rows)} (showing {len(shown)})\n" + "\n".join(
        f"- id={r['id']}, name={r['name']}, born={r['date_of_birth']}, course={r['course']}, email={r['email']}"
        for r in shown)
    prompt = ("You answer questions about a student database. Use ONLY the data below. "
              "If the answer is not in the data, say you don't know.\n\n"
              f"DATA:\n{context}\n\nQUESTION: {q.question}\nANSWER:")
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", timeout=120,
                       json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        r.raise_for_status()
        answer = r.json()["response"].strip()
    except httpx.HTTPError:
        raise HTTPException(503, f"Ollama is not reachable. Run `ollama serve` and `ollama pull {OLLAMA_MODEL}`.")
    return {"answer": answer, "matching_students": len(rows), "records_used": shown}


# ---------------------------------------------------------------- dummy demo data (first run only)
if os.getenv("SEED", "1") == "1" and not run("SELECT 1 FROM students LIMIT 1")[0]:
    for s in [("Aarav Sharma", "2004-03-12", "aarav@example.com", "9876543210", "Python"),
              ("Diya Patel", "2006-07-25", "diya@example.com", None, "Data Science"),
              ("Kabir Singh", "2005-11-02", "kabir@example.com", "9123456780", "Python"),
              ("Meera Nair", "2007-01-19", None, "9988776655", "Web Development"),
              ("Rohan Gupta", "2003-09-30", "rohan@example.com", None, "Data Science")]:
        run("INSERT INTO students (name,date_of_birth,email,phone,course) VALUES (?,?,?,?,?)", s)
