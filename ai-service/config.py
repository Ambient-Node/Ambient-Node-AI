"""설정 관리 (인식 기준 0.6으로 상향)"""
import os

class Config:
    TCP_IP = 'localhost'
    TCP_PORT = 8888
    
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 360
                        
    MQTT_SEND_INTERVAL = 1.0 / 6.0 # 1초에 6번 보내게 수정함
    FACE_ID_INTERVAL = 0.5
    FACE_LOST_TIMEOUT = 0.5
    MAX_MATCH_DISTANCE = 150
    MIN_FACE_SIZE = 0
    
    SIMILARITY_THRESHOLD = 0.7
    
    SAVE_DIR = os.getenv('SAVE_DIR', '/var/lib/ambient-node/captures')
    FACE_DIR = os.getenv('FACE_DIR', '/var/lib/ambient-node/users')
    MODEL_PATH = os.getenv('TFLITE_MODEL', '/app/facenet.tflite')
    
    BROKER = os.getenv('MQTT_BROKER', 'localhost')
    PORT = int(os.getenv('MQTT_PORT', 1883))