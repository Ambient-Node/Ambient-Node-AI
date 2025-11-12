import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import time
import os
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 저장 폴더 생성
save_dir = "captures"
os.makedirs(save_dir, exist_ok=True)

def enhance_image_for_detection(image):
    """얼굴 인식률 향상을 위한 이미지 전처리"""
    # 1. 밝기 및 대비 조정
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # 2. 가우시안 블러로 노이즈 제거
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

# FaceDetection 초기화 (장거리 모델 + 낮은 confidence)
with mp_face_detection.FaceDetection(
    model_selection=1,  # 0: 단거리(2m), 1: 장거리(5m) - 멀리서도 인식 가능
    min_detection_confidence=0.3) as face_detection:  # confidence 낮춤 (0.5 → 0.3)

    # **Fedora Linux용 카메라 초기화**
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 다른 카메라 인덱스를 시도합니다...")
        for i in range(1, 5):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"카메라 {i}번으로 연결되었습니다.")
                break
        else:
            print("사용 가능한 카메라를 찾을 수 없습니다.")
            exit()

    # **최적 해상도 설정**
    current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"현재 카메라 해상도: {current_width}x{current_height}")
    
    # FHD 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # **카메라 설정 최적화 (얼굴 인식률 향상)**
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 자동 노출 활성화
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # 자동 포커스 활성화
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"설정된 카메라 해상도: {actual_width}x{actual_height}")
    print("🔍 장거리 얼굴 인식 모드 (최대 5미터)")
    
    # 창 설정
    window_name = 'Enhanced Face Detection - Long Range'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    scale_factor = 0.8
    window_width = int(actual_width * scale_factor)
    window_height = int(actual_height * scale_factor)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    print(f"창 크기: {window_width}x{window_height}")
    print("조작 키:")
    print("  q: 종료")
    print("  c: 캡처")
    print("  +: 창 크기 확대")
    print("  -: 창 크기 축소")
    print("  r: 창 크기 리셋")
    print("  f: 전체화면 토글")
    print("  e: 이미지 향상 ON/OFF")

    current_scale = scale_factor
    enhancement_enabled = True  # 이미지 향상 기능 토글

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("이미지를 읽을 수 없습니다.")
            continue

        # BGR -> RGB 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # **이미지 향상 적용**
        if enhancement_enabled:
            processed_image = enhance_image_for_detection(image_rgb)
        else:
            processed_image = image_rgb

        # **다중 스케일 얼굴 감지** (더 작은 얼굴도 감지)
        results = face_detection.process(processed_image)
        
        # 원본 크기로 스케일 다운해서도 한번 더 시도 (멀리 있는 작은 얼굴 감지)
        if not results.detections:
            small_image = cv2.resize(processed_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            results_small = face_detection.process(small_image)
            if results_small.detections:
                # 스케일 보정
                for detection in results_small.detections:
                    # 좌표를 원본 크기로 변환
                    bboxC = detection.location_data.relative_bounding_box
                    bboxC.xmin /= 1.5
                    bboxC.ymin /= 1.5
                    bboxC.width /= 1.5
                    bboxC.height /= 1.5
                results = results_small

        # 상태 표시
        status_text = f"Enhancement: {'ON' if enhancement_enabled else 'OFF'} | Model: Long Range | Min Conf: 0.3"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 얼굴 그리기
        if results.detections:
            h, w, _ = frame.shape
            detection_count = len(results.detections)
            cv2.putText(frame, f"Faces Detected: {detection_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)

                # 인식 신뢰도(%)
                score = detection.score[0]
                confidence_text = f"{int(score*100)}%"

                # 얼굴 크기 계산 (거리 추정용)
                face_area = box_width * box_height
                distance_estimate = "Near" if face_area > 20000 else "Medium" if face_area > 8000 else "Far"
                
                # 얼굴 번호 및 거리 정보
                label = f"Person{i+1} ({confidence_text}) [{distance_estimate}]"

                # **거리에 따른 색상 변경**
                if distance_estimate == "Far":
                    color = (0, 0, 255)    # 빨간색 (멀리)
                elif distance_estimate == "Medium":
                    color = (0, 165, 255)  # 주황색 (중간)
                else:
                    color = (0, 255, 0)    # 초록색 (가까이)

                thickness = max(3, int(current_scale * 4))
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), 
                            color, thickness)

                # 텍스트 출력
                font_scale = max(0.6, current_scale * 1.0)
                text_thickness = max(2, int(current_scale * 3))
                cv2.putText(frame, label, (x_min, y_min - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
                
                # 박스 중앙 좌표 계산
                center_x = x_min + box_width // 2
                center_y = y_min + box_height // 2

                # 중앙점 표시 (거리에 따른 크기 조정)
                if distance_estimate == "Far":
                    circle_radius = max(3, int(current_scale * 5))
                elif distance_estimate == "Medium":
                    circle_radius = max(5, int(current_scale * 7))
                else:
                    circle_radius = max(7, int(current_scale * 9))
                    
                cv2.circle(frame, (center_x, center_y), circle_radius, (0, 0, 255), -1)

                # 좌표 및 면적 텍스트
                coord_font_scale = max(0.5, current_scale * 0.7)
                info_text = f"Center({center_x},{center_y}) Area:{face_area}"
                cv2.putText(frame, info_text,
                          (center_x + 20, center_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, coord_font_scale, (0, 0, 255), text_thickness)
        else:
            # 얼굴이 감지되지 않을 때
            cv2.putText(frame, "No faces detected - Try moving closer or adjusting lighting", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        # 키 입력 처리
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            timestamp = int(time.time())
            frame_filename = os.path.join(save_dir, f"enhanced_frame_{timestamp}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"📸 향상된 프레임 저장 완료: {frame_filename}")

            if results.detections:
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

                    face_filename = os.path.join(save_dir, f"enhanced_face_{timestamp}_{i}.png")
                    cv2.imwrite(face_filename, face_crop)
                    print(f"📸 향상된 얼굴 {i} 저장 완료: {face_filename}")
                    
        elif key == ord('e'):  # 이미지 향상 토글
            enhancement_enabled = not enhancement_enabled
            print(f"이미지 향상: {'활성화' if enhancement_enabled else '비활성화'}")
            
        elif key == ord('+') or key == ord('='):
            current_scale = min(1.5, current_scale + 0.1)
            new_width = int(actual_width * current_scale)
            new_height = int(actual_height * current_scale)
            cv2.resizeWindow(window_name, new_width, new_height)
            print(f"창 크기 확대: {new_width}x{new_height}")
            
        elif key == ord('-') or key == ord('_'):
            current_scale = max(0.3, current_scale - 0.1)
            new_width = int(actual_width * current_scale)
            new_height = int(actual_height * current_scale)
            cv2.resizeWindow(window_name, new_width, new_height)
            print(f"창 크기 축소: {new_width}x{new_height}")
            
        elif key == ord('r'):
            current_scale = 0.8
            new_width = int(actual_width * current_scale)
            new_height = int(actual_height * current_scale)
            cv2.resizeWindow(window_name, new_width, new_height)
            print(f"창 크기 리셋: {new_width}x{new_height}")
            
        elif key == ord('f'):
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("전체화면 해제")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("전체화면 모드")

    cap.release()
    cv2.destroyAllWindows()
    print("향상된 장거리 얼굴 인식 프로그램이 종료되었습니다.")
