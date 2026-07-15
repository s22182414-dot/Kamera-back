"""
Face Detection & Recognition Engine
Uses DeepFace (no C++ compilation required on Windows)
Backends: VGG-Face, Facenet512, ArcFace (selectable)
"""

import cv2
import numpy as np
import base64
import json
import os
from pathlib import Path
from typing import Optional
import database

# Suppress TF verbose output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class FaceEngine:
    def __init__(self):
        self.model_name = "VGG-Face"   # Options: VGG-Face, Facenet512, ArcFace
        self.detector_backend = "opencv"  # Fast: opencv | Accurate: retinaface
        self.known_persons: list[dict] = []
        self._deepface = None
        self._load_deepface()
        self.load_known_faces()

    def _load_deepface(self):
        """Lazy load DeepFace to avoid slow startup."""
        try:
            from deepface import DeepFace  # type: ignore
            self._deepface = DeepFace
            print("[FaceEngine] DeepFace loaded successfully")
        except Exception as e:
            print(f"[FaceEngine] WARNING: DeepFace not loaded: {e}")

    def load_known_faces(self):
        """Load known persons from database."""
        self.known_persons = database.get_persons()
        print(f"[FaceEngine] Loaded {len(self.known_persons)} known persons")

    def decode_image(self, image_base64: str) -> Optional[np.ndarray]:
        """Decode base64 image to numpy array (BGR for OpenCV)."""
        try:
            img_data = base64.b64decode(image_base64.split(",")[-1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[FaceEngine] Image decode error: {e}")
            return None

    def detect_faces(self, image_base64: str) -> dict:
        """Detect and recognize faces in image using DeepFace."""
        if self._deepface is None:
            return {"faces": [], "error": "DeepFace not loaded"}

        img = self.decode_image(image_base64)
        if img is None:
            return {"faces": [], "error": "Image decode failed"}

        try:
            # Detect faces
            face_objs = self._deepface.extract_faces(
                img_path=img,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )

            results = []
            for face_obj in face_objs:
                if face_obj.get("confidence", 0) < 0.5:
                    continue

                region = face_obj.get("facial_area", {})
                name = "Unknown"
                confidence = 0.0

                # Try to match against known persons
                if self.known_persons:
                    name, confidence = self._match_face(img, region)

                results.append({
                    "name": name,
                    "confidence": confidence,
                    "location": {
                        "top": region.get("y", 0),
                        "right": region.get("x", 0) + region.get("w", 0),
                        "bottom": region.get("y", 0) + region.get("h", 0),
                        "left": region.get("x", 0),
                    },
                    "is_known": name != "Unknown"
                })

            return {"faces": results, "count": len(results)}

        except Exception as e:
            print(f"[FaceEngine] Detection error: {e}")
            return {"faces": [], "error": str(e)}

    def _match_face(self, img: np.ndarray, region: dict) -> tuple[str, float]:
        """Match detected face against known persons."""
        if self._deepface is None:
            return ("Unknown", 0.0)
        try:
            # Crop face region
            x, y = region.get("x", 0), region.get("y", 0)
            w, h = region.get("w", 50), region.get("h", 50)
            face_crop = img[y:y+h, x:x+w]
            if face_crop.size == 0:
                return ("Unknown", 0.0)

            best_name = "Unknown"
            best_distance = 1.0

            for person in self.known_persons:
                photo_path = person.get("photo")
                if not photo_path or not Path(photo_path).exists():
                    continue
                try:
                    result = self._deepface.verify(
                        img1_path=face_crop,
                        img2_path=photo_path,
                        model_name=self.model_name,
                        detector_backend="skip",
                        enforce_detection=False
                    )
                    distance = result.get("distance", 1.0)
                    if distance < best_distance and result.get("verified", False):
                        best_distance = distance
                        best_name = person["name"]
                except Exception:
                    continue

            if best_name != "Unknown":
                # Convert distance to confidence %
                confidence = round((1 - best_distance) * 100, 1)
                return (best_name, confidence)
            return ("Unknown", 0.0)

        except Exception as e:
            print(f"[FaceEngine] Match error: {e}")
            return ("Unknown", 0.0)

    def extract_embedding(self, image_base64: str) -> Optional[list]:
        """Extract face embedding for registration."""
        if self._deepface is None:
            return None
        img = self.decode_image(image_base64)
        if img is None:
            return None
        try:
            embeddings = self._deepface.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            if embeddings:
                return embeddings[0].get("embedding", None)
            return None
        except Exception as e:
            print(f"[FaceEngine] Embedding error: {e}")
            return None

    def detect_motion(self, prev_frame_b64: str, curr_frame_b64: str) -> bool:
        """Detect motion between two frames using OpenCV frame differencing."""
        try:
            prev = self.decode_image(prev_frame_b64)
            curr = self.decode_image(curr_frame_b64)
            if prev is None or curr is None:
                return False

            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_pixels = np.sum(thresh > 0)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            motion_percent = (motion_pixels / total_pixels) * 100

            return motion_percent > 2.0  # >2% pixels changed = motion

        except Exception as e:
            print(f"[FaceEngine] Motion detect error: {e}")
            return False

# Singleton instance
face_engine = FaceEngine()
