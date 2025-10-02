# import sys
# import os

# # Context manager to suppress stderr temporarily
# class suppress_stderr:
#     def __enter__(self):
#         self._stderr = sys.stderr
#         sys.stderr = open(os.devnull, 'w')
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stderr.close()
#         sys.stderr = self._stderr

# import cv2
# import numpy as np
# import face_recognition

# # Suppress the warning when importing face_recognition_models
# with suppress_stderr():
#     import face_recognition_models  # only needed to avoid the warning

# # --- Load reference image ---
# imgModi = face_recognition.load_image_file('Images_Attendance/modi-image-for-InUth.jpg')
# imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)

# # --- Load test image ---
# imgTest = face_recognition.load_image_file('Images_Attendance/narendra-modi.jpg')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# # --- Encode reference face ---
# modi_encodings = face_recognition.face_encodings(imgModi)
# if len(modi_encodings) == 0:
#     raise ValueError("No face found in reference image!")
# encodeModi = modi_encodings[0]

# # Draw rectangle around reference face
# faceloc = face_recognition.face_locations(imgModi)[0]
# cv2.rectangle(imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

# # --- Encode test face ---
# test_encodings = face_recognition.face_encodings(imgTest)
# if len(test_encodings) == 0:
#     raise ValueError("No face found in test image!")
# encodeTest = test_encodings[0]

# # Draw rectangle around test face
# facelocTest = face_recognition.face_locations(imgTest)[0]
# cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

# # --- Compare faces ---
# results = face_recognition.compare_faces([encodeModi], encodeTest)
# faceDis = face_recognition.face_distance([encodeModi], encodeTest)
# print("Match:", results, "Distance:", faceDis)

# # Display result on test image
# cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50),
#             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

# # --- Show images ---
# cv2.imshow('Reference', imgModi)
# cv2.imshow('Test', imgTest)
# cv2.waitKey(0)
# cv2.destroyAllWindows()








import cv2
from deepface import DeepFace

# --- Absolute paths to your images ---
ref_path = r"C:\Users\Lenovo\Desktop\yuvraj projects\Face-recognition-Attendance-System-Project-main\Images_Attendance\narendra-modi.jpg"
test_path = r"C:\Users\Lenovo\Desktop\yuvraj projects\Face-recognition-Attendance-System-Project-main\Images_Attendance\narendra-modi.jpg"

# --- Load reference image ---
imgModi = cv2.imread(ref_path)
if imgModi is None:
    raise ValueError(f"Reference image not found at: {ref_path}")

# --- Load test image ---
imgTest = cv2.imread(test_path)
if imgTest is None:
    raise ValueError(f"Test image not found at: {test_path}")

# --- Detect faces using OpenCV Haar cascade ---
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Draw rectangle on reference image
gray_ref = cv2.cvtColor(imgModi, cv2.COLOR_BGR2GRAY)
ref_boxes = face_detector.detectMultiScale(gray_ref, 1.1, 4)
for (x, y, w, h) in ref_boxes:
    cv2.rectangle(imgModi, (x, y), (x + w, y + h), (155, 0, 255), 2)

# Draw rectangle on test image
gray_test = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
test_boxes = face_detector.detectMultiScale(gray_test, 1.1, 4)
for (x, y, w, h) in test_boxes:
    cv2.rectangle(imgTest, (x, y), (x + w, y + h), (155, 0, 255), 2)

# --- Compare faces using DeepFace ---
results = DeepFace.verify(
    img1_path=ref_path,
    img2_path=test_path,
    model_name='Facenet',
    detector_backend='opencv',
    enforce_detection=True
)

match = results['verified']
distance = results['distance']
print("Match:", match, "Distance:", distance)

# --- Put result text on test image ---
cv2.putText(imgTest, f'{match} {round(distance, 2)}', (50, 50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

# --- Show images ---
cv2.imshow('Reference Image', imgModi)
cv2.imshow('Test Image', imgTest)

print("Press ESC to close the windows.")
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
        break

cv2.destroyAllWindows()
