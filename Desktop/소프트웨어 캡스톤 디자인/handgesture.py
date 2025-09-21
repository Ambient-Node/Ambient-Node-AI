import cv2
import mediapipe as mp
import time

# Mediapipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 제스처 이름 정의 (원하는 제스처에 맞게 좌표 기준 설정)
def recognize_gesture(hand_landmarks, handedness_label):
    """
    손 랜드마크 기준으로 제스처 판정
    - Power : 손가락 모두 접음 (주먹)
    - Up    : 검지만 펴짐
    - Down  : 검지, 중지 펴짐
    - Rotate: 손가락 전부 펴짐
    """
    # 손가락 끝 landmark index
    tips = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 소지
    # 손가락 상태 저장 (1: 펴짐, 0: 접힘)
    fingers = []

    if handedness_label == "Right":
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # 나머지 손가락 판단 (y좌표 기준)
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # 제스처 판정
    if fingers == [0,0,0,0,0]:
        return "Power"
    elif fingers == [0,1,0,0,0]:
        return "Up"
    elif fingers == [0,1,1,0,0]:
        return "Down"
    elif fingers == [1,1,1,1,1]:
        return "Rotate"
    else:
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # 성능 향상: 프레임 크기 축소
            frame = cv2.resize(frame, (320, 240))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" 또는 "Right"
                    gesture = recognize_gesture(hand_landmarks, label)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(image, f"{gesture} ({label})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(image, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            cv2.imshow('Hand Gesture Recognition', image)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC로 종료
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
