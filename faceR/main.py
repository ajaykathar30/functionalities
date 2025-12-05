import os
import time
from pymongo import MongoClient
import face_recognition
import cv2
import numpy as np
from PIL import Image

# ---------- CONFIG ----------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "medical_db"
COLLECTION_NAME = "profiles"
FACES_DIR = "faces"   # where captured face images will be saved
CAPTURE_FILENAME = "captured_tmp.jpg"
CAPTURE_DELAY = 2     # seconds before capture to let user position
TOLERANCE = 0.4       # face_recognition tolerance, lower = stricter
# ----------------------------

def get_profiles_collection():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        return db[COLLECTION_NAME]
    except Exception as e:
        print("❌ MongoDB Connection Error:", e)
        return None

def capture_image(delay=CAPTURE_DELAY, filename=CAPTURE_FILENAME):
    """
    Captures a single frame from the default camera after `delay` seconds and saves it.
    Returns filepath or None on failure.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera. Check camera permissions or device.")
        return None

    print(f"Camera opened. Please position your face. Capturing in {delay} seconds...")
    time.sleep(delay)
    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        print("❌ Failed to capture image from camera.")
        return None

    # Save as JPEG
    cv2.imwrite(filename, frame)
    print(f"✅ Image captured and saved as: {filename}")
    return filename

def get_face_encoding_from_file(image_path):
    """
    Loads an image file and returns the first face encoding (list of floats) or None.
    """
    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        return None
    # convert numpy array to plain Python list of floats (so it's JSON/BSON serializable)
    return [float(x) for x in encodings[0]]

def fetch_all_known_encodings(collection):
    """
    Returns two lists: list_of_encodings (as lists) and list_of_docs (full docs).
    If no encodings found, returns ([], []).
    """
    encodings = []
    docs = []
    try:
        cursor = collection.find({})

        for doc in cursor:
            fe = doc.get("face_encoding", [])
            if fe and isinstance(fe, list) and len(fe) >= 128:
                encodings.append(np.array(fe, dtype=np.float64))
                docs.append(doc)
    except Exception as e:
        print("❌ Error fetching encodings from DB:", e)
    return encodings, docs

def print_medical_profile(doc):
    print("\n===== MATCHED PROFILE =====")
    print("Name:", doc.get("name", "Unknown"))
    mh = doc.get("medical_history", {})
    print("Allergies: ", ", ".join(mh.get("allergies", [])) if isinstance(mh.get("allergies", []), list) else mh.get("allergies", ""))
    print("Surgeries: ", ", ".join(mh.get("surgeries", [])) if isinstance(mh.get("surgeries", []), list) else mh.get("surgeries", ""))
    print("Health Conditions: ", ", ".join(mh.get("conditions", [])) if isinstance(mh.get("conditions", []), list) else mh.get("conditions", ""))
    print("===========================\n")

def insert_new_profile(collection, name, encoding_list, captured_image_path):
    """
    Insert a new profile document into MongoDB and save the face image under FACES_DIR.
    encoding_list: plain Python list of floats
    """
    # Ensure faces dir exists
    os.makedirs(FACES_DIR, exist_ok=True)
    # Save image as faces/<name>.jpg (avoid overwriting by appending timestamp if file exists)
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    target_path = os.path.join(FACES_DIR, f"{safe_name}.jpg")
    if os.path.exists(target_path):
        # append timestamp
        ts = int(time.time())
        target_path = os.path.join(FACES_DIR, f"{safe_name}_{ts}.jpg")
    try:
        # convert and save using PIL to ensure JPEG format
        img = Image.open(captured_image_path)
        img.convert("RGB").save(target_path, format="JPEG")
    except Exception:
        # fallback: copy file
        try:
            import shutil
            shutil.copy(captured_image_path, target_path)
        except Exception as e:
            print("⚠ Warning: could not save captured image to faces directory:", e)

    profile_doc = {
        "name": name,
        "face_encoding": encoding_list,
        "medical_history": {
            "allergies": [],    # will fill below
            "surgeries": [],
            "conditions": []
        }
    }

    # Ask structured questions (auto mode)
    print("\nPlease enter the medical details. If none, press Enter.")
    allergies = input("Enter allergies (comma separated): ").strip()
    surgeries = input("Enter surgeries (comma separated): ").strip()
    conditions = input("Enter health conditions (comma separated): ").strip()

    def parse_list_field(s):
        if not s:
            return []
        return [item.strip() for item in s.split(",") if item.strip()]

    profile_doc["medical_history"]["allergies"] = parse_list_field(allergies)
    profile_doc["medical_history"]["surgeries"] = parse_list_field(surgeries)
    profile_doc["medical_history"]["conditions"] = parse_list_field(conditions)

    try:
        res = collection.insert_one(profile_doc)
        print(f"\n✅ New profile inserted with _id: {res.inserted_id}")
        print(f"Face image saved at: {target_path}")
    except Exception as e:
        print("❌ Error inserting new profile into DB:", e)

def find_match_and_handle(collection):
    # 1) capture image
    captured = capture_image()
    if not captured:
        return

    # 2) get encoding from captured image
    unknown_encoding = get_face_encoding_from_file(captured)
    if unknown_encoding is None:
        print("⚠ No face detected in the captured image. Try again (ensure lighting and face is visible).")
        return

    # 3) fetch known encodings from DB
    known_encodings, docs = fetch_all_known_encodings(collection)

    if len(known_encodings) == 0:
        print("ℹ No known profiles in DB. Adding this person as a new profile.")
        name = input("Enter full name: ").strip()
        insert_new_profile(collection, name, unknown_encoding, captured)
        return

    # Convert unknown to numpy
    unk = np.array(unknown_encoding, dtype=np.float64)

    # 4) Compare using face_recognition: compute distances and matches
    try:
        distances = face_recognition.face_distance(known_encodings, unk)  # numpy arrays
        # filter matches below TOLERANCE
        valid_indices = [i for i, d in enumerate(distances) if d <= TOLERANCE]

        if len(valid_indices) == 0:
            # no valid match
            print("\nℹ No match found in DB.")
            c = input("Do you want to add this person to DB? (y/n): ").strip().lower()
            if c == 'y':
                name = input("Enter full name: ").strip()
                insert_new_profile(collection, name, unknown_encoding, captured)
            else:
                print("Skipped adding new profile.")
            return

        # pick closest among valid matches
        best_idx = valid_indices[np.argmin([distances[i] for i in valid_indices])]
        best_distance = float(distances[best_idx])
        matched_doc = docs[best_idx]

        print(f"\n✅ Match found (distance={best_distance:.4f}) for name: {matched_doc.get('name')}")
        print_medical_profile(matched_doc)

    except Exception as e:
        print("❌ Error while comparing faces:", e)
        print("\n No match found due to error.")
        return

if __name__ == "__main__":
    profiles_collection = get_profiles_collection()
    if profiles_collection is None:
        print("❌ Could not get collection. Exiting.")
        exit(1)

    print("✅ Connected to DB and 'profiles' collection is ready.")
    print("Press Enter to capture a photo and attempt recognition (or Ctrl+C to exit).")
    try:
        while True:
            input("Press Enter to start capture...")
            find_match_and_handle(profiles_collection)
    except KeyboardInterrupt:
        print("\nExiting. Goodbye.")
