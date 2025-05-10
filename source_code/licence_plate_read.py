from ultralytics import YOLO
import easyocr
import cv2

model = YOLO(r"C:\Users\DELL\OneDrive\Desktop\major project\runs\detect\license_plate_detection5\weights\best.pt")
image_path = (r"C:\Users\DELL\OneDrive\Desktop\major project\dataset\train\images\carLicence2_mp4-0054_jpg.rf.47bd000a6784f1ed797e702c7706fc18.jpg")
results = model(image_path)

boxes = results[0].boxes.xyxy.cpu().numpy()
img = cv2.imread(image_path)
reader = easyocr.Reader(['en'])

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    
    # Add margin
    margin = 10
    h, w, _ = img.shape
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    cropped = img[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_resized = cv2.resize(cropped_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    ocr_result = reader.readtext(cropped_resized)
    
    if ocr_result:
        for detection in ocr_result:
            print("Detected Text:", detection[1])
    else:
        print("No text detected.")


cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
