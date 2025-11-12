import cv2
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter

# -----------------------
# 설정
# -----------------------
face_dir = "faces_tflite"
os.makedirs(face_dir, exist_ok=True)

# TFLite 모델 로드
interpreter = Interpreter(model_path="/home/pi/projects/face/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

def get_embedding(face_img):
    """이미지에서 얼굴 임베딩 추출"""
    img = cv2.resize(face_img, tuple(input_shape))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding

def register_face_from_image(image_path, person_name):
    """PNG 이미지 파일에서 얼굴을 추출하여 등록"""
    
    # 이미지 로드
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return False
    
    print(f"[OK] Loaded image: {image_path}")
    print(f"[INFO] Image shape: {image.shape}")
    
    # 이미지를 얼굴 크기로 리사이즈
    face_img = image
    
    # 임베딩 추출
    try:
        embedding = get_embedding(face_img)
        print(f"[OK] Embedding extracted, shape: {embedding.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to extract embedding: {e}")
        return False
    
    # .npy 파일로 저장
    npy_path = os.path.join(face_dir, f"{person_name}.npy")
    np.save(npy_path, embedding)
    print(f"[OK] Face registered: {npy_path}")
    
    return True

if __name__ == "__main__":
    print("\n=== Face Registration ===\n")
    
    image_path = "./face-pngs/지윤.png"
    person_name = "지윤"
    
    print(f"Registering face from: {image_path}")
    print(f"Person name: {person_name}\n")
    
    if register_face_from_image(image_path, person_name):
        print(f"\n[SUCCESS] {person_name} registered successfully!")
        
        # 등록된 얼굴 확인
        print("\n=== Registered Faces ===")
        registered = sorted([f[:-4] for f in os.listdir(face_dir) if f.endswith(".npy")])
        for name in registered:
            print(f"  - {name}")
    else:
        print(f"\n[FAILED] Failed to register {person_name}")
