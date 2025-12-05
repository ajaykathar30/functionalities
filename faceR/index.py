from tabnanny import filename_only
import face_recognition
import cv2
import time
import os
from PIL import Image

def capture_image(delay=3, filename="captured_img.jpeg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return None
    
    print(f"Camera opened. Taking a picture in {delay} seconds...")
    time.sleep(delay)  # Wait for the specified delay

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"✅ Image captured and saved as {filename}")
        cap.release()
        cv2.destroyAllWindows()
        return filename
    else:
        print("Error: Could not capture image.")
        cap.release()
        cv2.destroyAllWindows()
        return None


def compareImages(encoding1, encoding2):
    print(encoding2)

    if len(encoding1) == 0 or len(encoding2) == 0:
        print("⚠ No face detected in one of the images.")
        return False
    else:
        results = face_recognition.compare_faces([encoding1[0]], encoding2[0])
        return results[0]


def findMatch(delay=3, captured_fname="captured_img.jpeg", folderpath="./familyMembers"):
    saved = capture_image(delay=delay, filename=captured_fname)
    if saved is None:
        print("Capture failed; aborting findMatch.")
        return None

    try:
        image1 = face_recognition.load_image_file(captured_fname)
    except Exception as e:
        print(f"Error loading captured image '{captured_fname}': {e}")
        return None

    encoding1 = face_recognition.face_encodings(image1)
    if len(encoding1) == 0:
        print("⚠ No face detected in the captured image.")
        return None

    if not os.path.isdir(folderpath):
        print(f"📁 Folder not found: {folderpath}")
        return None


    for filename in os.listdir(folderpath):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            continue

        image_path = os.path.join(folderpath, filename)
        try:
            image2 = face_recognition.load_image_file(image_path)
        except Exception as e:
            print(f"Skipping {filename}: cannot load ({e})")
            continue

        encoding2 = face_recognition.face_encodings(image2)

        try:
            is_match = compareImages(encoding1, encoding2)
        except Exception as e:
            print(f"Error comparing {filename}: {e}")
            continue

        if is_match:
            print(f"Match found: {filename.split('.')[0]}")
            return filename


    print(" No matching image found. Do you want to add new family member? (y/n)")
    c = input().strip().lower()

    if c == 'y':
        name = input("Enter your name: ").strip()

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

        new_path = os.path.join(folderpath, f"{name}.jpg")

        try:
            os.rename(captured_fname, new_path)
            print(f" New family member added and image saved as {name}.jpg")
        except Exception as e:
            print(f"❌ Error saving image: {e}")
    else:
        print("Bye bye")


  
findMatch()
