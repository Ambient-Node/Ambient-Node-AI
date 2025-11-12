#!/usr/bin/env python3
import paho.mqtt.client as mqtt

# MQTT 연결 성공 시 호출되는 콜백
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT 브로커에 연결되었습니다")
        # face/coords 토픽 구독
        client.subscribe("face/coords")
        print("face/coords 토픽 구독 시작")
    else:
        print(f"연결 실패, 코드: {rc}")

# 메시지 수신 시 호출되는 콜백
def on_message(client, userdata, message):
    try:
        # 메시지 디코딩
        coords_data = str(message.payload.decode("utf-8"))
        print(f"수신한 좌표: {coords_data}")
        
        # 좌표 파싱 (x, y ~ x, y 형식)
        if "~" in coords_data:
            # 두 개의 좌표 쌍으로 분리
            coord_pairs = coords_data.split("~")
            
            # 첫 번째 좌표
            first_coord = coord_pairs[0].strip().split(",")
            x1 = int(first_coord[0].strip())
            y1 = int(first_coord[1].strip())
            
            # 두 번째 좌표
            second_coord = coord_pairs[1].strip().split(",")
            x2 = int(second_coord[0].strip())
            y2 = int(second_coord[1].strip())
            
            print(f"첫 번째 좌표: ({x1}, {y1})")
            print(f"두 번째 좌표: ({x2}, {y2})")
            
    except Exception as e:
        print(f"메시지 처리 오류: {e}")

# MQTT 클라이언트 생성 (VERSION1 명시)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "face_coords_subscriber")

# 콜백 함수 등록
client.on_connect = on_connect
client.on_message = on_message

# 로컬 브로커에 연결 (localhost, 1883 포트)
try:
    client.connect("localhost", 1883, 60)
    print("MQTT 브로커 연결 시도 중...")
    
    # 메시지 수신 대기 (무한 루프)
    client.loop_forever()
    
except KeyboardInterrupt:
    print("\n프로그램 종료")
    client.disconnect()
except Exception as e:
    print(f"연결 오류: {e}")
