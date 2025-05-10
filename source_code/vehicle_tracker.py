import cv2
import easyocr
import pymysql
from ultralytics import YOLO
from datetime import datetime, timedelta
from rapidfuzz import fuzz
import traceback

print("Script started...")

# --- DB CONFIG ---
db_config = {
    'host': '127.0.0.1',
    'user': 'rakshita2002',
    'password': 'HelloMYSQL2002',
    'database': 'vehicle_tracker',
    'port': 3307,
    'connect_timeout': 10
}

# --- FILE PATHS ---
video_path = "c:/Users/DELL/OneDrive/Desktop/major project/carLicence1.mp4"
model_path = "c:/Users/DELL/OneDrive/Desktop/major project/runs/detect/license_plate_detection5/weights/best.pt"

# --- DATABASE CONNECT ---
print("Attempting to connect to database...")
try:
    db = pymysql.connect(**db_config)
    cursor = db.cursor()
    print("Connected to MySQL database.")
except pymysql.MySQLError as err:
    print(f"Database connection failed: {err}")
    traceback.print_exc()
    exit(1)

# Load YOLO model
print("Loading YOLOv8 model...")
try:
    model = YOLO(model_path)
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    raise

# Initialize OCR
print("Initializing EasyOCR...")
try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR initialized.")
except Exception as e:
    print(f"OCR init failed: {e}")
    raise

print("Opening video...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video at {video_path}")
    exit(1)
print("Video opened successfully.")

# Vehicle tracking
seen_plates = {}
EXIT_TIMEOUT = timedelta(seconds=10)
SIMILARITY_THRESHOLD = 85

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    frame_count += 1
    results = model(frame, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: No license plates detected.")
        cv2.imshow("Vehicle Tag Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break
        continue

    current_time = datetime.now()

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if conf < 0.3:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate_crop = frame[y1:y2, x1:x2]

        ocr_result = reader.readtext(plate_crop)
        for _, text, _ in ocr_result:
            plate_text = text.strip().upper()

            if not plate_text or len(plate_text) < 4:
                continue

            print(f"Frame {frame_count}: Detected plate → {plate_text}")

            found_match = False
            for seen_plate in list(seen_plates):
                similarity = fuzz.ratio(plate_text, seen_plate)
                if similarity >= SIMILARITY_THRESHOLD:
                    if current_time - seen_plates[seen_plate] > EXIT_TIMEOUT:
                        seen_plates[seen_plate] = current_time
                        try:
                            cursor.execute(
                                """
                                UPDATE vehicles 
                                SET exit_time = %s 
                                WHERE plate_number = %s 
                                AND exit_time IS NULL 
                                ORDER BY entry_time DESC 
                                LIMIT 1
                                """,
                                (current_time, seen_plate)
                            )
                            db.commit()
                            print(f"Exit logged: {seen_plate} at {current_time}")
                        except pymysql.MySQLError as err:
                            print(f"Update error: {err}")
                    found_match = True
                    break

            if not found_match:
                seen_plates[plate_text] = current_time
                try:
                    cursor.execute(
                        "INSERT INTO vehicles (plate_number, entry_time) VALUES (%s, %s)",
                        (plate_text, current_time)
                    )
                    db.commit()
                    print(f"Entry logged: {plate_text} at {current_time}")
                except pymysql.MySQLError as err:
                    print(f"Insert error: {err}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Vehicle Tag Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
print("Script complete. Resources released.")
