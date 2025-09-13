import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + MediaPipe 얼굴 검출 및 랜드마크")
    parser.add_argument(
        "--model",
        type=str,
        default="face_yolov8n.pt",
        help="YOLO face 모델 경로(.pt). 기본값: face_yolov8n.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="추론 디바이스 선택",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="신뢰도 임계값 (0~1)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="웹캠 인덱스 (기본 0)",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="화면에 FPS 표기",
    )
    return parser.parse_args()


class YOLOMediaPipeFaceDetector:
    def __init__(self, yolo_model_path):
        # YOLO 모델 로딩
        self.yolo_model = YOLO(yolo_model_path)
        
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_faces_and_landmarks(self, frame, conf_threshold=0.5):
        # YOLO로 얼굴 바운딩 박스 검출
        yolo_results = self.yolo_model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
        )
        
        boxes = []
        confs = []
        
        if yolo_results and len(yolo_results) > 0:
            r = yolo_results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                boxes = xyxy.tolist()
                confs = scores.tolist()
        
        # MediaPipe로 얼굴 랜드마크 검출
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.face_mesh.process(rgb_frame)
        
        landmarks = []
        if mp_results.multi_face_landmarks:
            for face_landmarks in mp_results.multi_face_landmarks:
                landmarks.append(face_landmarks)
        
        return boxes, confs, landmarks
    
    def draw_results(self, frame, boxes, confs, landmarks):
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # YOLO 바운딩 박스 그리기
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"face {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # MediaPipe 랜드마크 그리기
        for face_landmarks in landmarks:
            # 주요 랜드마크 인덱스 (MediaPipe Face Mesh 기준)
            # 코끝: 1, 좌안: 33, 우안: 362, 입 중앙: 13
            key_points = {
                'nose_tip': 1,
                'left_eye': 33,
                'right_eye': 362,
                'mouth_center': 13,
                'left_mouth': 61,
                'right_mouth': 291
            }
            
            colors = {
                'nose_tip': (0, 0, 255),      # 빨간색
                'left_eye': (255, 0, 0),      # 파란색
                'right_eye': (0, 255, 0),     # 초록색
                'mouth_center': (255, 255, 0), # 노란색
                'left_mouth': (255, 0, 255),  # 자주색
                'right_mouth': (0, 255, 255)  # 청록색
            }
            
            for name, idx in key_points.items():
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    
                    cv2.circle(output, (x, y), 3, colors[name], -1)
                    
                    # 코 좌표 텍스트 표시
                    if name == 'nose_tip':
                        cv2.putText(output, f"nose({x},{y})", (x + 5, y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return output


def main():
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print("[오류] YOLO 모델(.pt) 파일이 없습니다.")
        print("- 현재 경로:", Path.cwd())
        print("- 찾은 경로:", model_path.resolve())
        return
    
    print("YOLO + MediaPipe 얼굴 검출기 초기화 중...")
    detector = YOLOMediaPipeFaceDetector(str(model_path))
    
    print("웹캠 오픈 중... (종료: q)")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return
    
    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[오류] 프레임을 읽을 수 없습니다.")
                break
            
            # 얼굴 및 랜드마크 검출
            boxes, confs, landmarks = detector.detect_faces_and_landmarks(frame, args.conf)
            
            # 결과 그리기
            drawn = detector.draw_results(frame, boxes, confs, landmarks)
            
            # FPS 표시
            if args.show_fps:
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev_time))
                prev_time = now
                cv2.putText(drawn, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 얼굴 수 표시
            cv2.putText(drawn, f"faces: {len(boxes)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(drawn, f"landmarks: {len(landmarks)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("YOLO + MediaPipe Face Detection", drawn)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()