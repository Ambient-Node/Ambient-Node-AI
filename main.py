#!/usr/bin/env python3
"""AI Service - 1920x1080 입력 / 180도 회전 / 640 처리"""

import time
import cv2
import numpy as np
import mediapipe as mp
from config import Config
from camera import CameraStream
from face_recognition import FaceRecognizer
from face_tracker import FaceTracker
from mqtt_client import MQTTClient

# ✅ NMS: 중복 박스 제거
def non_max_suppression(boxes, scores, overlap_thresh=0.3):
    if len(boxes) == 0: return []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return pick

class AIService:
    def __init__(self, config):
        self.config = config
        self.camera = CameraStream(config)
        self.recognizer = FaceRecognizer(config.MODEL_PATH, config.FACE_DIR)
        
        # 트래커 초기화
        self.tracker = FaceTracker(
            max_distance=config.MAX_MATCH_DISTANCE,
            lost_timeout=config.FACE_LOST_TIMEOUT,
        )
        
        self.mqtt = MQTTClient(config.BROKER, config.PORT)
        self.mqtt.on_session_update = self.on_session_update
        self.mqtt.on_user_register = self.on_user_register
        self.mqtt.on_user_update = self.on_user_update
        self.mqtt.on_mode_change = self.on_mode_change
        
        self.current_mode = "manual_control"
        self.last_position_time = 0
        
        print(f"[AI] Init: Input 1920x1080 | Process {config.PROCESSING_WIDTH}x{config.PROCESSING_HEIGHT}")

    def on_mode_change(self, mode):
        print(f"[AI] Mode: {mode}")
        self.current_mode = mode
        if mode != 'ai_tracking':
            self.tracker.reset()
        
    def on_session_update(self, session_id, user_ids):
        self.recognizer.load_selected_users(user_ids)

    def on_user_register(self, payload):
        if payload.get('image_path'):
            self.recognizer.register_user(payload['user_id'], payload['username'], payload['image_path'])

    def on_user_update(self, payload):
        user_id = payload.get('user_id')
        if user_id in self.recognizer.known_usernames:
            self.recognizer.known_usernames[user_id] = payload.get('username')

    def run(self):
        print("[AI] Service Started (Auto-Rotate 180)")
        self.camera.start()
        
        # 전송 주기 (4Hz)
        target_send_interval = 0.25

        # 감지 설정 (model_selection=1: 원거리/전신용이 1920 해상도에서 더 적합할 수 있음)
        # 상황에 따라 0으로 변경 가능
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5 
        ) as face_detection:
            
            last_global_identify_time = 0
            
            try:
                while True:
                    if self.current_mode != 'ai_tracking':
                        time.sleep(1.0)
                        continue
                    
                    # 1. 원본 프레임 가져오기 (1920x1080)
                    frame = self.camera.get_frame()
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    
                    # frame = cv2.rotate(frame, cv2.ROTATE_180)
                    
                    current_time = time.time()
                    
                    # 2. 감지용 리사이즈 (640x360) - 16:9 비율 유지
                    frame_small = cv2.resize(frame, 
                        (self.config.PROCESSING_WIDTH, self.config.PROCESSING_HEIGHT))
                    
                    # 3. 얼굴 감지 수행 (NMS 적용됨) -> 결과는 1920x1080 좌표로 환산되어 나옴
                    detected_positions = self._detect_faces(frame_small, face_detection)
                    
                    # 4. 트래커 업데이트 (FHD 좌표 기준)
                    updated_ids, lost_faces = self.tracker.update(detected_positions, current_time)
                
                    # 5. 얼굴 신원 확인 (회전된 원본 FHD 프레임 사용 -> 인식률 최상)
                    force_identify = (current_time - last_global_identify_time >= self.config.FACE_ID_INTERVAL)
                    newly_identified = self.tracker.identify_faces(
                        self.recognizer, 
                        frame,
                        current_time,
                        interval=self.config.FACE_ID_INTERVAL,
                        force_all=force_identify
                    )
                    
                    if force_identify: last_global_identify_time = current_time
                    
                    for _, user_id, confidence in newly_identified:
                        self.mqtt.publish_face_detected(user_id, confidence)
                    
                    # 6. 좌표 전송 (4Hz)
                    if current_time - self.last_position_time >= target_send_interval:
                        session_id, selected_users = self.mqtt.get_current_session()
                        tracked_faces = self.tracker.get_selected_faces(selected_users)
                        
                        unique_users = {}
                        for finfo in tracked_faces:
                            # 유령 좌표 방지 (0.3초 컷)
                            if current_time - finfo['last_seen'] < 0.3:
                                unique_users[finfo['user_id']] = finfo
                        
                        for user_id, finfo in unique_users.items():
                            x, y = finfo['center']
                            self.mqtt.publish_face_position(user_id, x, y)
                        
                        self.last_position_time = current_time
                    
                    for lost_info in lost_faces:
                        self.mqtt.publish_face_lost(lost_info['user_id'], lost_info['duration'])
                    
                    time.sleep(0.001)
            
            except KeyboardInterrupt:
                pass
            finally:
                self.camera.stop()
                self.mqtt.stop()

    def _detect_faces(self, frame_processing, face_detection):
        # MediaPipe는 RGB 이미지를 원함
        rgb = cv2.cvtColor(frame_processing, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        
        detected = []
        if results.detections:
            boxes = []
            scores = []
            raw_detections = []
            
            # frame_processing은 640x360
            h_small, w_small, _ = frame_processing.shape

            for detection in results.detections:
                score = detection.score[0]
                bbox = detection.location_data.relative_bounding_box
                
                # NMS 계산용 (작은 화면 좌표)
                x1 = int(bbox.xmin * w_small)
                y1 = int(bbox.ymin * h_small)
                x2 = int((bbox.xmin + bbox.width) * w_small)
                y2 = int((bbox.ymin + bbox.height) * h_small)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                raw_detections.append(bbox)

            if len(boxes) > 0:
                # NMS 실행
                picked_indices = non_max_suppression(np.array(boxes), np.array(scores), overlap_thresh=0.3)

                for i in picked_indices:
                    bbox_raw = raw_detections[i]
                    
                    # ✅ [핵심 변환] 상대좌표(0.0~1.0) * 원본해상도(1920x1080)
                    orig_w = self.config.CAMERA_WIDTH
                    orig_h = self.config.CAMERA_HEIGHT
                    
                    x_center = int((bbox_raw.xmin + bbox_raw.width / 2) * orig_w)
                    y_center = int((bbox_raw.ymin + bbox_raw.height / 2) * orig_h)
                    
                    x1_orig = int(bbox_raw.xmin * orig_w)
                    y1_orig = int(bbox_raw.ymin * orig_h)
                    x2_orig = int((bbox_raw.xmin + bbox_raw.width) * orig_w)
                    y2_orig = int((bbox_raw.ymin + bbox_raw.height) * orig_h)

                    detected.append({
                        'center': (x_center, y_center),
                        'bbox': (x1_orig, y1_orig, x2_orig, y2_orig) # 원본 FHD 좌표
                    })
        
        return detected

def main():
    config = Config()
    service = AIService(config)
    service.run()

if __name__ == '__main__':
    main()