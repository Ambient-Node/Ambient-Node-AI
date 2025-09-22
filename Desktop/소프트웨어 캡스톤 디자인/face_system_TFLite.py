import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image

import tensorflow as tf

# -----------------------
# 설정
# -----------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

save_dir = "captures"
face_dir = "faces_tflite"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(face_dir, exist_ok=True)

FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

# -----------------------
# 이미지 향상 함수
# -----------------------
def enhance_image_for_detection(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced

# -----------------------
# 한글 텍스트 출력 함수
# -----------------------
def put_text_kor(frame, text, pos, color=(0,255,0), font_size=30):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -----------------------
# Cosine similarity
# -----------------------
def cosine_similarity(a, b):
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

# -----------------------
# TFLite FaceNet 모델 로드
# -----------------------
interpreter = tf.lite.Interpreter(model_path="facenet.tflite")  # 모델 경로
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

# -----------------------
# 등록된 얼굴 벡터 불러오기
# -----------------------
def load_known_faces():
    known_embeddings = []
    known_names = []
    for file in os.listdir(face_dir):
        if file.lower().endswith((".npy")):
            emb = np.load(os.path.join(face_dir, file))
            name = os.path.splitext(file)[0]
            known_embeddings.append(emb)
            known_names.append(name)
    print(f"✅ 등록된 얼굴 수: {len(known_names)} → {known_names}")
    return known_embeddings, known_names

known_embeddings, known_names = load_known_faces()

# -----------------------
# 얼굴 임베딩 추출
# -----------------------
def get_embedding(face_img):
    img = cv2.resize(face_img, tuple(input_shape))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0  # [-1,1]
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding

# -----------------------
# 실시간 얼굴 인식
# -----------------------
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as face_detection:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        for i in range(1,5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"카메라 {i}번 연결 성공")
                break
        else:
            print("사용 가능한 카메라 없음")
            exit()

    window_name = "Face Detection + TFLite FaceNet"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    enhancement_enabled = True
    scale_factor = 0.8
    current_scale = scale_factor

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = enhance_image_for_detection(image_rgb) if enhancement_enabled else image_rgb
        results = face_detection.process(processed_image)

        detected_faces = []

        if results.detections:
            h, w, _ = frame.shape
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)

                face_crop = frame[
                    max(0, y_min):min(h, y_min + box_height),
                    max(0, x_min):min(w, x_min + box_width)
                ]

                detected_faces.append((x_min, y_min, box_width, box_height, face_crop))

                # 임베딩 추출
                embedding = get_embedding(face_crop)

                # Cosine similarity로 인식
                name = "Unknown"
                confidence = 0.0
                if known_embeddings:
                    sims = [cosine_similarity(embedding, k_emb) for k_emb in known_embeddings]
                    best_idx = np.argmax(sims)
                    if sims[best_idx] > 0.5:  # 신뢰도 기준
                        name = known_names[best_idx]
                        confidence = sims[best_idx]

                # 얼굴 크기 기반 거리 추정
                face_area = box_width * box_height
                distance_estimate = "Far" if face_area < 8000 else "Medium" if face_area < 20000 else "Near"

                # 박스 색상
                color = (0,0,255) if distance_estimate=="Far" else (0,165,255) if distance_estimate=="Medium" else (0,255,0)
                thickness = max(2,int(current_scale*3))
                cv2.rectangle(frame, (x_min, y_min), (x_min+box_width, y_min+box_height), color, thickness)

                # 중앙점 표시
                center_x = x_min + box_width//2
                center_y = y_min + box_height//2
                circle_radius = max(5,int(current_scale*7))
                cv2.circle(frame, (center_x, center_y), circle_radius, (0,0,255), -1)

                # 이름 + 신뢰도 표시
                label = f"{name} ({confidence*100:.1f}%) [{distance_estimate}]"
                frame = put_text_kor(frame, label, (x_min, y_min-25), color=color, font_size=int(20*current_scale))

        # 상태 표시
        status_text = f"Enhancement: {'ON' if enhancement_enabled else 'OFF'} | Faces: {len(results.detections) if results.detections else 0}"
        cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        # -------------------------
        # 키 이벤트
        # -------------------------
        if key == ord('q'):
            break
        elif key == ord('c'):
            timestamp = int(time.time())
            cv2.imwrite(os.path.join(save_dir, f"frame_{timestamp}.png"), frame)
            print(f"📸 프레임 저장 완료: frame_{timestamp}.png")
        elif key == ord('e'):
            enhancement_enabled = not enhancement_enabled
            print(f"이미지 향상: {'활성화' if enhancement_enabled else '비활성화'}")
        elif key == ord('r'):
            if detected_faces:
                for i,(x,y,w_,h_,crop) in enumerate(detected_faces):
                    cv2.imshow(f"Register Face {i+1}", crop)
                    new_name = input(f"등록할 이름 (Face {i+1}): ").strip()
                    if new_name:
                        emb = get_embedding(crop)
                        np.save(os.path.join(face_dir,f"{new_name}.npy"), emb)
                        print(f"✅ 얼굴 등록 완료: {new_name}")
                        known_embeddings, known_names = load_known_faces()

    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")
