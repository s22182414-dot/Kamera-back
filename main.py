import os
import json
import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

import database as db
from face_engine import face_engine

app = FastAPI(title="CamAI Backend", version="1.0.0")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve stored photos/thumbnails statically
db.init_db()
app.mount("/data", StaticFiles(directory="data"), name="data")


# ─── Pydantic Models ──────────────────────────────
class EventCreate(BaseModel):
    type: str  # "motion", "face_known", "face_unknown"
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    image_base64: Optional[str] = None

class PersonCreate(BaseModel):
    name: str
    role: str
    photo_base64: str

class ChatRequest(BaseModel):
    message: str
    date: Optional[str] = None

class SettingsUpdate(BaseModel):
    cameraResolution: Optional[str] = None
    cameraFps: Optional[int] = None
    motionSensitivity: Optional[float] = None
    faceConfidence: Optional[int] = None
    geminiApiKey: Optional[str] = None
    soundAlerts: Optional[bool] = None


# ─── API Endpoints ─────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "CamAI Server", "time": datetime.datetime.now().isoformat()}

# ─── Events API ────────────────────────────────────
@app.get("/api/events")
def get_events_list(date: Optional[str] = None, limit: int = 100):
    events = db.get_events(date=date, limit=limit)
    return {"status": "success", "count": len(events), "events": events}

@app.post("/api/events")
def create_event(payload: EventCreate):
    event = db.add_event(
        event_type=payload.type,
        person_name=payload.person_name,
        confidence=payload.confidence,
        image_base64=payload.image_base64
    )
    return {"status": "success", "event": event}

@app.delete("/api/events")
def clear_all_events():
    db.save_db({"events": [], "persons": db.get_persons()})
    return {"status": "success", "message": "All events cleared"}

@app.get("/api/stats/today")
def get_today_stats():
    summary = db.get_today_summary()
    return summary


# ─── Persons API ───────────────────────────────────
@app.get("/api/persons")
def get_persons_list():
    persons = db.get_persons()
    return {"status": "success", "count": len(persons), "persons": persons}

@app.post("/api/persons")
def create_person(payload: PersonCreate):
    # Register person in DB
    person = db.add_person(
        name=payload.name,
        face_encoding=[],
        photo_base64=payload.photo_base64
    )
    # Reload known faces in face recognition engine
    face_engine.load_known_faces()
    return {"status": "success", "person": person}

@app.delete("/api/persons/{person_id}")
def delete_person(person_id: int):
    database_data = db.get_db()
    database_data["persons"] = [p for p in database_data.get("persons", []) if p.get("id") != person_id]
    db.save_db(database_data)
    face_engine.load_known_faces()
    return {"status": "success", "message": f"Person {person_id} deleted"}


# ─── AI DeepFace Recognition API ───────────────────
@app.post("/api/recognize")
def recognize_face(payload: EventCreate):
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    result = face_engine.recognize(payload.image_base64)
    return {"status": "success", "result": result}


# ─── AI Chat / Gemini Synthesis API ────────────────
@app.post("/api/chat")
def ai_chat(payload: ChatRequest):
    target_date = payload.date or datetime.date.today().isoformat()
    events = db.get_events(date=target_date, limit=100)

    known_list = [e for e in events if e.get("type") == "face_known"]
    unknown_count = len([e for e in events if e.get("type") == "face_unknown"])
    motion_count = len([e for e in events if e.get("type") == "motion"])

    names = list(set([e.get("person_name") for e in known_list if e.get("person_name")]))

    reply = f"📅 {target_date} sanasidagi kamera yozuvlari tahlili:\n" \
            f"• Jami voqealar: {len(events)} ta\n" \
            f"• Taniqli shaxslar: {len(known_list)} ta ({', '.join(names) if names else 'yoq'})\n" \
            f"• Noma'lum shaxslar: {unknown_count} ta\n" \
            f"• Harakatlar: {motion_count} ta"

    return {
        "status": "success",
        "date": target_date,
        "reply": reply,
        "events_count": len(events)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
