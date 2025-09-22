import cv2
import mediapipe as mp
import time
import os
import numpy as np
import face_recognition
from PIL import ImageFont, ImageDraw, Image

# -----------------------
# 설정
# -----------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

save_dir = "captures"
face_dir = "faces"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(face_dir, exist_ok=True)

# 한글 폰트 경로 (macOS)
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
# 등록된 얼굴 불러오기
# -----------------------
def load_known_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir(face_dir):
        path = os.path.join(face_dir, file)
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(os.path.splitext(file)[0])
    print(f"✅ 등록된 얼굴 수: {len(known_names)} → {known_names}")
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# -----------------------
# 실시간 얼굴 인식
# -----------------------
with mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.4) as face_detection:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        for i in range(1,5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"카메라 {i}번 연결 성공")
                break
        else:
            print("사용 가능한 카메라 없음")
        # exit() 대신 while 루프에서 계속 시도 가능

    window_name = "Face Detection + Recognition"
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

                # 기본 이름
                name = "Unknown"
                if face_crop.size > 0:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_crop)
                    if encodings:
                        match = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.45)
                        face_distances = face_recognition.face_distance(known_encodings, encodings[0])
                        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                        if best_match_index is not None and match[best_match_index]:
                            name = known_names[best_match_index]

                # 얼굴 크기 기반 거리 추정
                face_area = box_width * box_height
                distance_estimate = "Far" if face_area < 8000 else "Medium" if face_area < 20000 else "Near"

                # 박스 색상
                if distance_estimate == "Far":
                    color = (0, 0, 255)
                elif distance_estimate == "Medium":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                thickness = max(2, int(current_scale*3))
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), color, thickness)

                # 중앙점 표시
                center_x = x_min + box_width // 2
                center_y = y_min + box_height // 2
                circle_radius = max(5, int(current_scale*7))
                cv2.circle(frame, (center_x, center_y), circle_radius, (0, 0, 255), -1)

                # 신뢰도 표시
                score = detection.score[0]
                confidence_text = f"{int(score*100)}%"
                label = f"{name} ({confidence_text}) [{distance_estimate}]"

                frame = put_text_kor(frame, label, (x_min, y_min - 25), color=color, font_size=int(20*current_scale))

                info_text = f"Center({center_x},{center_y}) Area:{face_area}"
                frame = put_text_kor(frame, info_text, (center_x + 20, center_y), color=(0,0,255), font_size=int(15*current_scale))

        # 상태 표시
        status_text = f"Enhancement: {'ON' if enhancement_enabled else 'OFF'} | Faces: {len(results.detections) if results.detections else 0}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(30) & 0xFF
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

                    # 사용자 선택: 1-새 등록, 2-이름 변경, 3-취소
                    action = input(f"Face {i+1}: 1-등록, 2-이름 변경, 3-취소: ").strip()

                    if action == "1":  # 새 얼굴 등록
                        new_name = input(f"등록할 이름 (Face {i+1}): ").strip()
                        if new_name:
                            # -------------------------
                            # 중복 검사 및 삭제
                            # -------------------------
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            encodings = face_recognition.face_encodings(rgb_crop)
                            if encodings:
                                enc = encodings[0]
                                to_delete = []
                                for idx, known_enc in enumerate(known_encodings):
                                    dist = face_recognition.face_distance([known_enc], enc)[0]
                                    if dist < 0.4:  # 동일 인물
                                        print(f"⚠️ 기존 얼굴 '{known_names[idx]}' 삭제 후 등록")
                                        to_delete.append(known_names[idx])
                                for name_del in to_delete:
                                    path_del = os.path.join(face_dir, f"{name_del}.jpg")
                                    if os.path.exists(path_del):
                                        os.remove(path_del)
                                        print(f"🗑️ 삭제 완료: {path_del}")

                            # -------------------------
                            # 새로운 얼굴 저장
                            # -------------------------
                            filename = os.path.join(face_dir, f"{new_name}.jpg")
                            cv2.imwrite(filename, crop)
                            print(f"✅ 얼굴 등록 완료: {filename}")

                    elif action == "2":  # 이름 변경
                        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        encodings = face_recognition.face_encodings(rgb_crop)
                        if encodings:
                            enc = encodings[0]
                            # 가장 가까운 기존 얼굴 찾기
                            distances = face_recognition.face_distance(known_encodings, enc)
                            if len(distances) > 0:
                                idx = np.argmin(distances)
                                old_name = known_names[idx]
                                new_name = input(f"'{old_name}'의 새 이름: ").strip()
                                if new_name:
                                    old_path = os.path.join(face_dir, f"{old_name}.jpg")
                                    new_path = os.path.join(face_dir, f"{new_name}.jpg")
                                    if os.path.exists(old_path):
                                        os.rename(old_path, new_path)
                                        print(f"🔄 '{old_name}' → '{new_name}' 변경 완료")
                                        # 박스에 표시되는 이름 갱신
                                        known_encodings, known_names = load_known_faces()

                    else:  # 취소
                        print("등록/이름 변경 취소")

                # 등록/변경 후 인코딩 재로드
                known_encodings, known_names = load_known_faces()


    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")
