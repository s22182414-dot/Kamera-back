import os
import json
import base64
import datetime
from pathlib import Path

DB_PATH = Path("data/events.json")
PERSONS_PATH = Path("data/persons")
EVENTS_PATH = Path("data/events")

def init_db():
    """Initialize database directories and files."""
    PERSONS_PATH.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        DB_PATH.write_text(json.dumps({"events": [], "persons": []}, ensure_ascii=False, indent=2))

def get_db():
    """Read database."""
    if not DB_PATH.exists():
        init_db()
    return json.loads(DB_PATH.read_text(encoding="utf-8"))

def save_db(data: dict):
    """Save database."""
    DB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def add_event(event_type: str, person_name: str = None, image_base64: str = None, confidence: float = None):
    """Add new event to database."""
    db = get_db()
    now = datetime.datetime.now()

    event = {
        "id": len(db["events"]) + 1,
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "type": event_type,  # "motion", "face_known", "face_unknown"
        "person_name": person_name,
        "confidence": confidence,
        "image": None
    }

    # Save image file
    if image_base64:
        img_filename = f"event_{event['id']}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        img_path = EVENTS_PATH / img_filename
        img_data = base64.b64decode(image_base64.split(",")[-1])
        img_path.write_bytes(img_data)
        event["image"] = str(img_path)

    db["events"].append(event)
    save_db(db)
    return event

def get_events(date: str = None, limit: int = 50):
    """Get events, optionally filtered by date."""
    db = get_db()
    events = db["events"]

    if date:
        events = [e for e in events if e.get("date") == date]

    # Sort by timestamp descending
    events = sorted(events, key=lambda x: x["timestamp"], reverse=True)
    return events[:limit]

def get_persons():
    """Get all known persons."""
    db = get_db()
    return db.get("persons", [])

def add_person(name: str, face_encoding: list, photo_base64: str = None):
    """Add known person to database."""
    db = get_db()

    person = {
        "id": len(db["persons"]) + 1,
        "name": name,
        "face_encoding": face_encoding,
        "added_at": datetime.datetime.now().isoformat(),
        "photo": None
    }

    if photo_base64:
        photo_filename = f"person_{person['id']}_{name.replace(' ', '_')}.jpg"
        photo_path = PERSONS_PATH / photo_filename
        img_data = base64.b64decode(photo_base64.split(",")[-1])
        photo_path.write_bytes(img_data)
        person["photo"] = str(photo_path)

    db["persons"].append(person)
    save_db(db)
    return person

def get_today_summary():
    """Get summary of today's events."""
    today = datetime.date.today().isoformat()
    events = get_events(date=today, limit=200)

    known_visitors = {}
    unknown_count = 0

    for event in events:
        if event["type"] == "face_known" and event["person_name"]:
            if event["person_name"] not in known_visitors:
                known_visitors[event["person_name"]] = []
            known_visitors[event["person_name"]].append(event["time"])
        elif event["type"] == "face_unknown":
            unknown_count += 1

    return {
        "date": today,
        "total_events": len(events),
        "known_visitors": known_visitors,
        "unknown_count": unknown_count,
        "events": events[:20]
    }
