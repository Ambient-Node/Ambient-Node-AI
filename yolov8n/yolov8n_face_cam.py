import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n-face 실시간 얼굴 검출 (웹캠)")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-face.pt",
        help="YOLOv8 face 모델 경로(.pt). 기본값: yolov8n-face.pt",
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
        "--imgsz",
        type=int,
        default=640,
        help="추론 입력 이미지 크기 (정사각)",
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


def draw_boxes_with_center(frame: np.ndarray, boxes, confidences) -> np.ndarray:
    output = frame.copy()
    
    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 바운딩 박스 그리기
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 바운딩 박스 중앙 좌표 계산
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 중앙에 빨간 점 그리기
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)  # 빨간색 원
        
        # 중앙 좌표 텍스트 표시
        coord_text = f"center({center_x},{center_y})"
        cv2.putText(output, coord_text, (center_x + 10, center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 신뢰도 라벨
        label = f"face {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(output, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return output


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print("[오류] yolov8n-face 모델(.pt) 파일이 없습니다.")
        print("- 현재 경로:", Path.cwd())
        print("- 찾은 경로:", model_path.resolve())
        print("- 해결: yolov8n-face.pt 파일을 프로젝트 폴더에 넣거나 --model 경로를 지정하세요.")
        return

    print("YOLOv8n-face 모델 로딩 중...", model_path)
    try:
        # Torch 2.6+ weights_only 정책으로 인한 호환 이슈는 Ultralytics 내부에서 처리됩니다.
        model = YOLO(str(model_path))
    except Exception as e:
        print("[오류] 모델 로딩 실패:", e)
        return

    print(f"디바이스: {args.device}")
    print("웹캠 오픈 중... (종료: q)")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다. 다른 인덱스를 시도하거나 권한을 확인하세요.")
        return

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[오류] 프레임을 읽을 수 없습니다.")
                break

            # 추론
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            boxes = []
            confs = []
            
            if results and len(results) > 0:
                r = results[0]
                
                # 바운딩 박스 정보
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    boxes = xyxy.tolist()
                    confs = scores.tolist()

            drawn = draw_boxes_with_center(frame, boxes, confs)

            if args.show_fps:
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev_time))
                prev_time = now
                cv2.putText(drawn, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(drawn, f"faces: {len(boxes)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("YOLOv8n-face Webcam", drawn)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

