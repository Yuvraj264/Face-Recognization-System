import cv2
import os
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
IMAGE_FOLDER = r"C:\Users\Lenovo\Desktop\yuvraj projects\Face-recognition-Attendance-System-Project-main\Images_Attendance"
ATTENDANCE_FILE = "Attendance.csv"
RECOGNIZED_FOLDER = "RecognizedFaces"
UNKNOWN_FOLDER = "UnknownFaces"  # New folder to auto-capture unknowns
SIMILARITY_THRESHOLD = 0.35
FRAME_SKIP = 3  # dynamic frame skip
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# ---------------- INITIALIZE ----------------
reference_embeddings = []
names = []
attendance_today = set()

os.makedirs(RECOGNIZED_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ---------------- LOAD REFERENCE FACES ----------------
def load_references():
    global reference_embeddings, names
    reference_embeddings = []
    names = []

    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png"))]
    for f in files:
        path = os.path.join(IMAGE_FOLDER, f)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        reference_embeddings.append(faces[0].embedding)
        names.append(os.path.splitext(f)[0].upper())
    print(f"Loaded {len(reference_embeddings)} reference faces.")

load_references()

# ---------------- ATTENDANCE FUNCTION ----------------
def mark_attendance(name, face_img):
    if name not in attendance_today:
        attendance_today.add(name)
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "w") as f:
                f.write("Name,Time,Date\n")
        with open(ATTENDANCE_FILE, "a") as f:
            now = datetime.now()
            f.writelines(f"{name},{now.strftime('%H:%M:%S')},{now.strftime('%d/%m/%Y')}\n")

        # Save recognized face screenshot
        save_path = os.path.join(RECOGNIZED_FOLDER, f"{name}_{now.strftime('%H%M%S')}.jpg")
        cv2.imwrite(save_path, face_img)

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- CLAHE for low-light ----------------
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

# ---------------- START WEBCAM ----------------
cap = cv2.VideoCapture(0)
frame_count = 0
unknown_count = 0

print("Webcam started. ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    # Enhance low-light
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Detect faces
    faces = app.get(processed_frame)
    for face in faces:
        x1, y1, x2, y2 = [int(v*2) for v in face.bbox]
        emb = face.embedding
        face_img = frame[y1:y2, x1:x2]

        # Compare with known references
        similarities = [cosine_similarity(emb, ref_emb) for ref_emb in reference_embeddings]
        max_sim = max(similarities) if similarities else 0
        best_idx = np.argmax(similarities) if similarities else -1
        name = "Unknown"

        if max_sim >= SIMILARITY_THRESHOLD:
            name = names[best_idx]
            color = (0, 255, 0)
            mark_attendance(name, face_img)
        else:
            # Automatically save unknown face
            color = (0, 0, 255)
            unknown_count += 1
            save_path = os.path.join(UNKNOWN_FOLDER, f"unknown_{unknown_count}_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(save_path, face_img)

        # Draw box and name
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} ({max_sim:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Live attendance overlay
    overlay_start_y = 30
    for idx, student in enumerate(sorted(attendance_today)):
        cv2.putText(frame, f"{student}", (10, overlay_start_y + idx*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance session ended.")
