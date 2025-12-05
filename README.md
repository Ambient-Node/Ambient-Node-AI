
***

# Ambient-Node-AI

## 1. 개요.

Ambient-Node-AI 레포지토리는 AI 비전 기반 사용자 추적형 스마트 에어서큘레이터의 얼굴 인식·추적 기능을 담당하는 엣지 AI 모듈을 포함함.  
Raspberry Pi 5 상에서 TensorFlow Lite, OpenCV, MediaPipe 기반 파이프라인을 구성하고, 추론 결과를 MQTT 및 UART 프로토콜을 통해 팬 제어 시스템과 연동함.  

- 대상 플랫폼: Raspberry Pi 5 + Pi Camera Module V3.  
- 주요 라이브러리: tflite-runtime, OpenCV, NumPy, MediaPipe(검출), MQTT 클라이언트.  
- 핵심 기능: 얼굴 검출 → 임베딩 추출 → 사용자 인식 → 얼굴 추적 → 팬 제어 명령 전송.  

문자 개념도.

- [카메라 입력] → [AI 비전 파이프라인] → [MQTT 좌표 메시지] → [Fan 서비스] → [UART] → [MCU 제어]  

***

## 2. AI 비전 파이프라인.

### 2.1 Dual-Resolution 전략.

시스템은 실시간성과 인식 정확도를 동시에 확보하기 위해 해상도 이원화(Dual-Resolution) 파이프라인을 사용함.  

- 저해상도 스트림 (예: 640x360).  
  - 용도: 얼굴 검출(Detection).  
  - 효과: 연산량 감소 및 FPS 향상.  

- 고해상도 스트림 (FHD, 1920x1080).  
  - 용도: 임베딩 추출 및 인식(Recognition).  
  - 역할: 검출된 얼굴 좌표를 고해상도 좌표계로 변환해 고품질 얼굴 크롭 확보.  

문자 도식.

- 입력: FHD 프레임  
  - ├─ 다운스케일 → [D-스트림: 얼굴 검출]  
  - └─ 원본 유지 → [R-스트림: FHD 얼굴 크롭 → 임베딩 추론 → 사용자 인식]  

이 구조를 통해 별도 서버 없이 엣지 디바이스 단독으로 고해상도 얼굴 인식과 실시간 추적을 동시에 만족함.  

***

## 3. 얼굴 인식 모듈 (FaceRecognizer).

`face_recognition.py`는 TensorFlow Lite 기반 얼굴 임베딩 추출 및 코사인 유사도 기반 사용자 인식을 담당하는 핵심 모듈임.  

### 3.1 모델 초기화 및 입력 전처리.

- TFLite 모델 로딩.  
  - Interpreter(model_path=...) 초기화 후 allocate_tensors 호출.  
  - get_input_details, get_output_details로 입력 텐서의 높이, 너비 등 메타데이터 획득.  

- 얼굴 이미지 전처리.  
  - 입력 얼굴 ROI를 모델 입력 크기로 리사이즈.  
  - float32 형으로 변환.  
  - 각 픽셀에 (x - 127.5) / 128.0 연산을 적용해 값 범위를 약 -1에서 1 사이로 조정.  
  - 배치 차원을 추가해 (1, H, W, C) 텐서 형태로 변환 후 Interpreter 입력.  

문자 흐름도.

- [얼굴 ROI] → 리사이즈 → float32 변환 → 정규화 연산 → 배치 차원 추가 → [TFLite 추론]  

### 3.2 임베딩 추출 및 정규화.

모델 추론 결과로 얻은 임베딩 벡터에 L2 정규화를 적용해 방향 정보만 남기고 크기에 따른 영향을 제거함.  

- 임베딩 벡터의 길이를 계산해 0보다 크면, 각 성분을 길이로 나누어 크기가 1인 단위 벡터로 변환.  
- 이렇게 만든 단위 임베딩은 서로 간 유사도를 내적만으로 계산할 수 있는 형태가 됨.  

문자 도식.

- [원본 임베딩] → 벡터 길이 계산 → 각 성분 / 길이 → [정규화 임베딩]  

정규화된 임베딩은 사용자 간 분포를 안정적으로 분리하고, 임계값 기반 분류 로직 설계에 유리한 특성 제공.  

### 3.3 코사인 유사도 기반 인식.

등록된 사용자 임베딩과 현재 얼굴 임베딩 사이의 코사인 유사도를 계산해 가장 유사한 사용자를 선택함.  

- 내부 저장 구조.  
  - known_embeddings: 정규화된 등록 사용자 임베딩 리스트.  
  - known_user_ids: 각 임베딩에 대응되는 사용자 ID 리스트.  

- 인식 절차.  
  - 입력 얼굴에서 임베딩을 생성하고 정규화.  
  - known_embeddings와의 행렬 내적을 수행해 유사도 벡터 sims 계산.  
  - sims에서 최대값 인덱스를 찾고, 해당 값이 threshold(기본 0.6)보다 크면 해당 user_id 반환.  

- 임계값 운영.  
  - threshold를 높이면 오인식이 줄어들고, 낮추면 인식 민감도가 증가.  

문자 도형.

- [등록 임베딩 집합 E]  
- [현재 얼굴 임베딩 v]  
- [각 e와 v의 내적 → sims]  
- [최대 sims 선택 후 threshold와 비교] → 조건 만족 시 사용자 인식 성공  

### 3.4 사용자 등록 및 프로필 관리.

`register_user`는 정적인 얼굴 이미지에서 임베딩을 생성하고 파일 시스템에 저장하는 기능 담당.  

- 입력.  
  - user_id: 내부 식별용 ID.  
  - username: 사용자 이름.  
  - image_path: 등록용 얼굴 이미지 경로.  

- 처리 절차.  
  - 이미지 로딩 후 임베딩 생성.  
  - face_dir/user_id/embedding.npy 파일에 임베딩 저장.  
  - metadata.json에 user_id, username 정보를 JSON 형식으로 저장.  

- 선택 사용자 로딩.  
  - load_selected_users(user_ids) 호출 시, 해당 ID 목록에 대한 embedding.npy만 읽어 메모리에 로딩.  
  - 유효한 임베딩만 다시 정규화해 known_embeddings, known_user_ids에 반영.  

디렉터리 구조.

- faces/  
  - └── <user_id>/  
    - ├── embedding.npy  
    - └── metadata.json  

이 구조를 통해 사용자 추가, 삭제, 비활성화 작업을 파일 단위로 단순하게 수행하고, 선택 사용자만 로딩해 메모리 사용량과 추론 시간을 절약함.  

***

## 4. 얼굴 추적 모듈 (FaceTracker).

`face_tracker.py`는 프레임 간 얼굴 위치를 추적해 안정적인 face_id를 유지하는 경량 트래킹 모듈임.  

### 4.1 기본 개념과 상태 구조.

FaceTracker는 외부 검출 모듈이 제공하는 얼굴 위치 리스트를 입력으로 받아, 이전 프레임의 추적 대상과 연결하는 상위 레벨 추적기임. 자체 검출은 수행하지 않음.  

- 주요 상태.  
  - tracked_faces: face_id를 키로 가지는 딕셔너리.  
    - center: 얼굴 중심 좌표 (cx, cy).  
    - bbox: 얼굴 경계 상자 (x1, y1, x2, y2).  
    - user_id: 인식된 사용자 ID 또는 None.  
    - confidence: 최근 인식 신뢰도.  
    - last_seen: 마지막 검출 시각.  
    - last_identified: 마지막 인식 시각.  
    - first_seen: 처음 관측된 시각.  
  - max_distance: 같은 얼굴로 인정할 최대 거리(픽셀 단위).  
  - lost_timeout: 이 시간 이상 관측되지 않으면 추적 종료로 판단.  

멀티 스레드 환경 안전성을 위해 내부적으로 Lock을 사용해 상태를 보호함.  

### 4.2 유클리드 거리 기반 매칭.

update(detected_positions, current_time)는 현재 프레임의 검출 결과 리스트를 기반으로 기존 트래커를 갱신하거나 새 트래커를 생성함.  

- Greedy 매칭 알고리즘 요약.  
  - 입력 리스트의 각 검출에 대해 `_find_closest(center)` 호출.  
  - `_find_closest`는 모든 tracked_faces의 center와의 거리 값을 계산.  
  - 가장 가까운 트래커의 거리값이 max_distance 미만이면 그 face_id에 연결하고, 아니면 새로운 face_id 생성.  

- 만료 처리.  
  - `_remove_expired(current_time, timeout)`에서 current_time - last_seen이 timeout을 초과하는 face_id 제거.  
  - 제거 시 user_id가 존재하면 lost_faces 목록에 user_id와 duration 정보를 기록해 반환.  

문자 도식.

- 프레임 t 기준.  
  - 검출 얼굴: [F1, F2, ...]  
  - 기존 트래커: [T1, T2, ...]  

- 각 Fi에 대해.  
  - 모든 Tj와의 거리 계산.  
  - 최소 거리이면서 max_distance 이하인 Tj에 연결.  
  - 조건 만족 트래커가 없으면 새 face_id 생성.  

연산량이 상대적으로 작고 구현이 단순해, 라즈베리파이 기반 엣지 환경에 적합한 추적 방식임.  

### 4.3 얼굴 인식 연동 및 재인식 정책.

identify_faces(recognizer, frame, current_time, interval, force_all=False)는 추적 중인 얼굴에 대해 필요한 시점에만 인식을 수행하도록 설계된 함수임.  

- 인식 대상 선택.  
  - current_time - last_identified가 interval 이상인 얼굴만 인식 대상에 포함.  
  - force_all이 true이면 모든 트래커에 대해 인식 수행.  

- 처리 절차.  
  - 각 face_id의 bbox로 FHD 프레임에서 얼굴 ROI를 크롭.  
  - ROI가 비어 있지 않으면 recognizer.recognize(face_crop)를 호출.  
  - user_id가 반환되면 해당 트래커에 user_id, confidence, last_identified, first_seen 등 상태 갱신.  
  - 동일 user_id가 연속으로 인식될 경우 confidence를 최대 0.95까지 단계적으로 상향 조정해 신뢰도 강화.  

- 선택 사용자 필터링.  
  - get_selected_faces(selected_user_ids)는 tracked_faces 중 user_id가 selected_user_ids 목록에 포함되는 항목만 반환.  
  - 이 결과를 이용해 팬 제어 대상 사용자를 한정.  

문자 흐름.

- [검출 결과] → FaceTracker.update  
- [주기적인 타이머] → identify_faces 호출  
- [인식 결과(user_id, confidence)] → 추적 상태 갱신  
- [선택 사용자 필터링] → Fan 서비스로 좌표 전달  

***

## 5. AI·제어 통합 아키텍처.

### 5.1 마이크로서비스 및 이벤트 흐름.

AI 모듈은 Docker 기반 마이크로서비스 구조 내에서 독립 컨테이너로 실행되며, MQTT 브로커를 통해 다른 서비스와 데이터를 교환함.  

- AI 컨테이너.  
  - 카메라 스트림 수신.  
  - 얼굴 검출 및 FaceTracker 업데이트.  
  - 필요 시 FaceRecognizer를 호출해 사용자 인식.  
  - 추적 좌표와 user_id를 MQTT 토픽으로 발행.  

- Fan 서비스.  
  - ambient/ai/face-position 토픽을 구독.  
  - 얼굴 중심 좌표를 팬 헤드의 목표 각도로 변환.  
  - UART 명령(P x,y 형식 등)으로 Xiao RP2040 MCU에 전달.  

- BLE Gateway.  
  - 모바일 앱에서 선택된 사용자 ID, 운전 모드 변경 등의 요청을 MQTT로 전달.  
  - AI 모듈과 Fan 서비스가 동일한 상태를 공유하도록 중계.  

문자 흐름도.

- [Camera]  
  → [AI 컨테이너: 검출·인식·추적]  
  → [MQTT: face-position]  
  → [Fan 컨테이너: 좌표 → 각도 변환]  
  → [UART] → [Xiao RP2040] → [모터 제어]  

### 5.2 실시간성 및 안정성 설계.

- 추적 및 인식 주기.  
  - FaceTracker 업데이트는 매 프레임마다 수행.  
  - FaceRecognizer 호출은 interval 기반으로 제한하여 CPU 부하 관리.  

- 제어 반응 특성.  
  - 라즈베리파이와 제어 보드 간 UART, PI 제어, Soft Start/Stop 알고리즘을 조합해 평균 약 20ms 수준 초기 반응 지연과 약 180ms 이내 목표 위치 도달.  

- 데이터 발행 주기.  
  - 얼굴 좌표 MQTT 발행 주기를 약 250ms로 설정하여, 팬이 과도하게 진동하지 않으면서 부드럽게 사용자를 따라가도록 구성.  

이와 같은 이벤트 기반 구조 덕분에 AI, 제어, 앱, DB 간 결합도가 낮고, 다른 센서나 인식 모델을 추가할 때도 최소 수정으로 확장 가능함.  

***

## 6. AI 기술적 특징 및 의의.

- 엣지 온디바이스 얼굴 인식.  
  - 클라우드 서버 없이 Raspberry Pi 단일 보드에서 얼굴 검출, 임베딩 생성, 사용자 인식을 모두 수행.  
  - 네트워크 연결이 불안정하거나 차단된 환경에서도 자율 추적 기능 유지.  

- 코사인 유사도 + 정규화 기반 경량 인식.  
  - L2 정규화된 임베딩과 코사인 유사도 비교만으로 사용자 인식을 수행해, 추가적인 분류 레이어 없이도 사용자를 확장 가능한 구조.  
  - 사용자 수가 증가해도 단순 내적 연산만으로 동작하므로 구현이 단순하고 추론 비용 예측이 용이.  

- 거리 기반 트래킹의 실용적 활용.  
  - 복잡한 딥러닝 추적기 대신, 유클리드 거리 기반 Greedy 매칭으로 구현 난이도와 연산량을 낮추면서 실내 1~2명 환경에서 충분한 추적 성능 확보.  

- AI·제어 통합 엣지 모션 플랫폼.  
  - 얼굴 인식 결과를 곧바로 2축 팬 제어에 연결하는 엔드 투 엔드 파이프라인 구현으로, 지능형 CCTV, 펫캠, 서비스 로봇 등 다양한 도메인으로 확장 가능한 모션 플랫폼 기반 마련.  

***
