import cv2
import mediapipe as mp
import numpy as np
import os
import time
import struct
import threading
import argparse
from tflite_runtime.interpreter import Interpreter
import paho.mqtt.client as mqtt
from datetime import datetime
from collections import deque

# ========================================
# 설정 변수 (최상단)
# ========================================
CAMERA_WIDTH = 1920  # FHD
CAMERA_HEIGHT = 1080  # FHD
DISPLAY_WIDTH = 1920  # FHD
DISPLAY_HEIGHT = 1080  # FHD
PROCESSING_WIDTH = 640  # 얼굴 인식용 해상도
PROCESSING_HEIGHT = 360
MQTT_SEND_INTERVAL = 0.25
FACE_IDENTIFICATION_INTERVAL = 1.0
MIN_FACE_SIZE = 800

# 경로 설정
SAVE_DIR = "/home/pi/projects/softcapstone/captures"
FACE_DIR = "/home/pi/projects/softcapstone/faces_tflite"
MODEL_PATH = "/home/pi/projects/face/facenet.tflite"

# MQTT 설정
BROKER = "localhost"
MQTT_PORT = 1883
TOPIC = "face/coords"

# USB 웹캠 설정
USB_CAMERA_INDEX = 0

# ========================================
# 명령줄 인자 파싱
# ========================================
parser = argparse.ArgumentParser(description='Face Recognition System')
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display)')
parser.add_argument('--display', action='store_true', help='Run with display window')
args = parser.parse_args()

# 헤드리스 모드 결정
if args.headless:
    HEADLESS_MODE = True
elif args.display:
    HEADLESS_MODE = False
else:
    # 옵션이 없으면 사용자에게 물어봄
    print("\n=== Face Recognition System ===")
    print("Select display mode:")
    print("  0 = Show display window")
    print("  1 = Headless mode (no display)")
    while True:
        try:
            mode = int(input("Enter mode (0 or 1): ").strip())
            if mode in [0, 1]:
                HEADLESS_MODE = (mode == 1)
                break
            else:
                print("[ERROR] Please enter 0 or 1")
        except ValueError:
            print("[ERROR] Please enter a valid number")

print(f"[OK] Running in {'HEADLESS' if HEADLESS_MODE else 'DISPLAY'} mode\n")

# ========================================
# 초기화
# ========================================
mp_face_detection = mp.solutions.face_detection

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

# MQTT 연결
client = mqtt.Client()
try:
    client.connect(BROKER, MQTT_PORT, 60)
    print(f"[OK] MQTT broker connected: {BROKER}:{MQTT_PORT}")
except Exception as e:
    print(f"[ERROR] MQTT connection failed: {e}")

# USB 웹캠 초기화
cap = cv2.VideoCapture(USB_CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("[ERROR] Failed to open USB webcam")
    exit(1)

print(f"[OK] USB webcam opened (index: {USB_CAMERA_INDEX})")

# 프레임 큐
frame_queue = deque(maxlen=1)
queue_lock = threading.Lock()

# ========================================
# USB 웹캠 캡처 스레드
# ========================================
def webcam_capture():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Webcam frame capture failed")
            time.sleep(0.1)
            continue
        with queue_lock:
            frame_queue.append(frame)
        time.sleep(0.005)

# 캡처 스레드 시작
capture_thread = threading.Thread(target=webcam_capture, daemon=True)
capture_thread.start()
print("[OK] Webcam capture thread started")

# ========================================
# 유틸리티 함수
# ========================================
def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ========================================
# TFLite 모델 로드
# ========================================
print("[INFO] Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]
print("[OK] TFLite model loaded")

# ========================================
# 등록된 얼굴 로드
# ========================================
def load_known_faces():
    known_embeddings = []
    known_names = []
    for file in os.listdir(FACE_DIR):
        if file.lower().endswith(".npy"):
            emb = np.load(os.path.join(FACE_DIR, file))
            name = os.path.splitext(file)[0]
            known_embeddings.append(emb)
            known_names.append(name)
    print(f"[OK] Loaded {len(known_names)} registered faces: {known_names}")
    return known_embeddings, known_names

known_embeddings, known_names = load_known_faces()

# ========================================
# 임베딩 추출 함수
# ========================================
def get_embedding(face_img):
    img = cv2.resize(face_img, tuple(input_shape))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding

# ========================================
# 메인 루프
# ========================================
print(f"[INFO] Starting face detection ({'headless' if HEADLESS_MODE else 'with display'})...")

# 얼굴 추적 정보
tracked_faces = {}
next_face_id = 0

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
    last_send_time = time.time()
    last_identification_time = time.time()
    frame_count = 0
    fps_start = time.time()
    fps = 0.0
    
    scale_x = DISPLAY_WIDTH / PROCESSING_WIDTH
    scale_y = DISPLAY_HEIGHT / PROCESSING_HEIGHT

    if not HEADLESS_MODE:
        window_name = "Face Detection + TFLite FaceNet (FHD)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    print("[INFO] Waiting for frames...")

    try:
        while True:
            with queue_lock:
                if not frame_queue:
                    time.sleep(0.001)
                    continue
                frame = frame_queue[0]

            frame_display = frame.copy() if not HEADLESS_MODE else None
            frame_processing = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
            image_rgb = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            current_time = time.time()
            detected_face_positions = []
            
            # 1. 얼굴 감지
            if results.detections:
                h, w, _ = frame_processing.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * w)
                    y_min = int(bboxC.ymin * h)
                    box_width = int(bboxC.width * w)
                    box_height = int(bboxC.height * h)

                    face_area = box_width * box_height
                    if face_area < MIN_FACE_SIZE:
                        continue

                    x_min_fhd = int(x_min * scale_x)
                    y_min_fhd = int(y_min * scale_y)
                    box_width_fhd = int(box_width * scale_x)
                    box_height_fhd = int(box_height * scale_y)
                    center_x_fhd = x_min_fhd + box_width_fhd//2
                    center_y_fhd = y_min_fhd + box_height_fhd//2

                    detected_face_positions.append({
                        "bbox": (x_min, y_min, box_width, box_height),
                        "bbox_fhd": (x_min_fhd, y_min_fhd, box_width_fhd, box_height_fhd),
                        "center": (center_x_fhd, center_y_fhd)
                    })

            # 2. 얼굴 추적
            updated_face_ids = set()
            for face_pos in detected_face_positions:
                center = face_pos["center"]
                closest_id = None
                min_distance = float('inf')
                
                for face_id, face_info in tracked_faces.items():
                    old_center = face_info["center"]
                    distance = ((center[0] - old_center[0])**2 + (center[1] - old_center[1])**2)**0.5
                    if distance < min_distance and distance < 300:
                        min_distance = distance
                        closest_id = face_id
                
                if closest_id:
                    tracked_faces[closest_id]["bbox_fhd"] = face_pos["bbox_fhd"]
                    tracked_faces[closest_id]["center"] = face_pos["center"]
                    tracked_faces[closest_id]["last_seen"] = current_time
                    tracked_faces[closest_id]["bbox_processing"] = face_pos["bbox"]
                    updated_face_ids.add(closest_id)
                else:
                    tracked_faces[next_face_id] = {
                        "name": "Unidentified",
                        "confidence": 0.0,
                        "bbox_fhd": face_pos["bbox_fhd"],
                        "center": face_pos["center"],
                        "last_seen": current_time,
                        "last_identified": 0.0,
                        "bbox_processing": face_pos["bbox"]
                    }
                    updated_face_ids.add(next_face_id)
                    next_face_id += 1

            # 오래된 얼굴 제거
            expired_ids = [fid for fid, finfo in tracked_faces.items() 
                          if current_time - finfo["last_seen"] > 2.0]
            for fid in expired_ids:
                del tracked_faces[fid]

            # 3. 얼굴 신원 확인 (1초마다)
            if current_time - last_identification_time >= FACE_IDENTIFICATION_INTERVAL:
                print(f"[DEBUG] Identifying {len(updated_face_ids)} faces")
                
                for face_id in updated_face_ids:
                    if face_id not in tracked_faces:
                        continue
                    
                    face_info = tracked_faces[face_id]
                    x_min, y_min, box_width, box_height = face_info["bbox_processing"]
                    
                    face_crop = frame_processing[
                        max(0, y_min):min(PROCESSING_HEIGHT, y_min + box_height),
                        max(0, x_min):min(PROCESSING_WIDTH, x_min + box_width)
                    ]

                    if face_crop.size == 0:
                        continue

                    embedding = get_embedding(face_crop)
                    name = "Unknown"
                    confidence = 0.0
                    
                    if known_embeddings:
                        sims = [cosine_similarity(embedding, k_emb) for k_emb in known_embeddings]
                        best_idx = np.argmax(sims)
                        if sims[best_idx] > 0.4:
                            name = known_names[best_idx]
                            confidence = sims[best_idx]
                    
                    tracked_faces[face_id]["name"] = name
                    tracked_faces[face_id]["confidence"] = confidence
                    tracked_faces[face_id]["last_identified"] = current_time
                
                last_identification_time = current_time

            # 4. 알려진 얼굴만 수집
            known_face_centers = []
            known_face_infos = []
            
            for face_id, face_info in tracked_faces.items():
                if face_info["name"] != "Unknown" and face_info["name"] != "Unidentified":
                    known_face_centers.append(face_info["center"])
                    known_face_infos.append({
                        "name": face_info["name"],
                        "confidence": face_info["confidence"],
                        "x": face_info["center"][0],
                        "y": face_info["center"][1],
                        "bbox": face_info["bbox_fhd"]
                    })

            # 화면 표시
            if not HEADLESS_MODE:
                for face_id, face_info in tracked_faces.items():
                    x_min, y_min, box_width, box_height = face_info["bbox_fhd"]
                    center_x, center_y = face_info["center"]
                    
                    if face_info["name"] == "Unidentified":
                        continue
                    elif face_info["name"] == "Unknown":
                        color = (0, 165, 255)
                        label = "Unknown"
                    else:
                        color = (0, 255, 0)
                        label = f"{face_info['name']} ({face_info['confidence']*100:.1f}%)"
                    
                    cv2.rectangle(frame_display, (x_min, y_min), (x_min+box_width, y_min+box_height), color, 3)
                    cv2.circle(frame_display, (center_x, center_y), 8, (0, 0, 255), -1)
                    cv2.putText(frame_display, label, (x_min, y_min-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                status_text = f"FPS: {fps:.1f} | Tracked: {len(tracked_faces)} | Known: {len(known_face_infos)}"
                cv2.putText(frame_display, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                cv2.imshow(window_name, frame_display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break

            # MQTT 전송
            if time.time() - last_send_time >= MQTT_SEND_INTERVAL and known_face_centers:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n[{timestamp}] === DETECTED KNOWN FACES ===")
                for info in known_face_infos:
                    print(f"  - Name: {info['name']:<10} | Confidence: {info['confidence']*100:>5.1f}% | Position: ({info['x']}, {info['y']})")
                csv_data = ";".join([f"{x},{y}" for (x, y) in known_face_centers])
                binary_data = b"".join([struct.pack("ii", x, y) for (x, y) in known_face_centers])
                client.publish(TOPIC, csv_data)
                client.publish(TOPIC+"_bin", binary_data)
                print(f"[{timestamp}] MQTT transmitted")
                last_send_time = time.time()

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                fps = 30 / elapsed
                fps_start = time.time()
                print(f"[INFO] FPS: {fps:.1f} | Tracked: {len(tracked_faces)} | Known: {len(known_face_infos)}")

    except KeyboardInterrupt:
        print("\n[INFO] Terminating program...")
    finally:
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        cap.release()
        print("[INFO] Program terminated")
