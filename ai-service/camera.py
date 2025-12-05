import cv2
import numpy as np
import socket
import threading
import time
from collections import deque # 앞,뒤로 데이터를 넣고 빼는 속도가 엄청 빠른 queue


class CameraStream:
    def __init__(self, config):
        self.config = config
        self.frame_queue = deque(maxlen=1) # 큐의 크기를 1개로 제한해서 새 데이터가 들어오면 옛날 데이터는 자동으로 밀려남.
        self.lock = threading.Lock() # 여러 쓰레드가 동시에 데이터에 손대지 못하게 막는 자물쇠. 잠깐 문을 잠글 때 씀.
        self.running = False


    def start(self):
        """카메라 스트림 시작"""
        print(f"[Camera] Using external rpicam-vid at tcp://{self.config.TCP_IP}:{self.config.TCP_PORT}")
        self.running = True
        threading.Thread(target=self._receive_stream, daemon=True).start() # _receive_stream 스레드 생성해서 메인 프로그램에서 제외함.


    def _receive_stream(self):
        """TCP 스트림 수신 (최적화), 영상 수신 함수"""
        max_retries = 10
        retry_delay = 3
        
        for attempt in range(max_retries): # 접속 시도 반복문
            try:
                # 1. 소켓(전화기) 만들기. AF_INET(인터넷), SOCK_STREAM(TCP 방식)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)  # 2MB 버퍼
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 지연 최소화 : 딜레이 없이 바로바로 보내라고 설정. Nagle 알고리즘 끄기
                
                sock.settimeout(5) # 5초동안 응답 없으면 끊음
                
                # 실제 연결 시도 (IP, PORT)
                sock.connect((self.config.TCP_IP, self.config.TCP_PORT))
                print(f"[Camera] Connected to {self.config.TCP_IP}:{self.config.TCP_PORT}")
                break
            except (ConnectionRefusedError, socket.timeout) as e:
                print(f"[Camera] Connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"[Camera] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print("[Camera] Max retries reached, stream unavailable")
                    self.running = False
                    return
        
        sock.settimeout(None)  # 블로킹 모드
        
        buffer = b""
        frame_size = self.config.CAMERA_WIDTH * self.config.CAMERA_HEIGHT * 3 // 2  # YUV420 포맷은 RGB의 1.5배 크기


        while self.running:
            try:
                # 1. 네트워크에서 chunk를 128kb씩 받음.
                chunk = sock.recv(131072)
                if not chunk:
                    print("[Camera] Stream ended")
                    self.running = False
                    break
                
                buffer += chunk
                
                # 버퍼에 한 장의 사진 분량이 모였는지 확인
                while len(buffer) >= frame_size:
                    # 딱 한 장 분량만 잘라냄
                    frame_data = buffer[:frame_size]
                    # 남은 찌꺼기는 다시 버퍼에 저장함
                    buffer = buffer[frame_size:]
                    
                    try:
                        # 데이터 변환 - byte -> 숫자 행렬 -> 이미지(YUV)
                        yuv = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                            (self.config.CAMERA_HEIGHT * 3 // 2, self.config.CAMERA_WIDTH)
                        )
                        
                        # 색상 변환 YUV -> BGR
                        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                        
                        # 완성된 이미지를 큐에 넣음
                        # with self.lock 은 카메라 쓰레드는
                        with self.lock:
                            self.frame_queue.append(bgr)
                    except Exception:
                        continue
            
            except Exception as e:
                print(f"[Camera] ❌ Frame receive error: {e}")
                break


        sock.close()
        print("[Camera] Receiver stopped")


    def get_frame(self):
        """가장 최근 프레임 반환 (복사 없음)"""
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1]
        return None


    def stop(self):
        self.running = False
