                                           🧑‍💻 Face Recognition Attendance System

📖 Project Overview
A real-time face recognition attendance system that automatically marks attendance using a webcam.
It identifies known faces using InsightFace embeddings and DeepFace verification, saves recognized faces, and captures unknown faces automatically for later reference.

Perfect for classrooms, offices, or events where manual attendance is cumbersome.



⚡ Features:-

👀 Real-time face recognition via webcam
📝 Automatic attendance logging in CSV format
🚫 Auto-capture of unknown faces
📸 Store screenshots of recognized faces
🌙 Low-light enhancement with CLAHE
📊 Similarity check using cosine similarity



🛠️ Tech Stack:-

🐍 Python 3.11
📷 OpenCV – image and video processing
🧠 InsightFace – face embeddings for high accuracy
🤖 DeepFace – face verification
🔢 NumPy – numerical computations
📄 CSV – attendance logging



🗂️ Project Structure
Face-recognition-Attendance-System-Project/
│
├─ 🖥️ attendanceproject.py      # Real-time attendance system with webcam
├─ 🔍 main.py                   # Face verification using DeepFace
├─ 🐍 venv/                     # Python virtual environment
├─ 🖼️ Images_Attendance/        # Reference images of known people
├─ ❓ UnknownFaces/             # Automatically captured unknown faces
├─ ✅ RecognizedFaces/          # Screenshots of recognized people
├─ 📝 Attendance.csv            # CSV file storing attendance records



🚀 Installation & Setup:-

Clone the repository:
git clone <your-repo-link>
cd Face-recognition-Attendance-System-Project



Create virtual environment and activate it:-

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate


Install dependencies:-

pip install -r requirements.txt



Prepare folders:-

Images_Attendance/ → Add reference images (JPEG/PNG).
UnknownFaces/ → Automatically stores unknown faces.
RecognizedFaces/ → Automatically stores recognized face screenshots.



Run the attendance system:-

python attendanceproject.py
Optional: Verify a face manually using DeepFace
python main.py



🖼️ Usage & Live Attendance Overlay:-

Launch the system with your webcam.
Faces detected in real-time:
✅ Recognized faces → Green rectangle, name, similarity score
❌ Unknown faces → Automatically saved in UnknownFaces/ folder
Attendance logged in Attendance.csv with Name, Time, Date



Example Live Overlay Visualization:-
------------------------------
| Attendance Today:           |
| 1. YUVRAJ                   |
| 2. NANDHA                   |
| 3. MODI                     |
------------------------------
[Green Box] Recognized Face
[Red Box]   Unknown Face



💡 Future Enhancements:-

📧 Add email or notification alerts for unknown faces
🖥️ Integrate with a GUI dashboard for live statistics
🎥 Support multiple cameras and large-scale classrooms
🛡️ Add role-based recognition for different types of users
📸 Screenshots



📄 License:-

This project is licensed under the MIT License.



📋 requirements.txt:-

opencv-python>=4.7.0
numpy>=1.24.0
insightface>=0.7.3
deepface>=0.0.89
# Optional for advanced features
matplotlib>=3.7.0
pandas>=2.1.0
