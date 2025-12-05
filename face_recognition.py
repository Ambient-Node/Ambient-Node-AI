"""얼굴 인식 모델 관리 (코사인 유사도 + 벡터 정규화 적용)"""
import os
import cv2
import json
import numpy as np
from tflite_runtime.interpreter import Interpreter

class FaceRecognizer:
    def __init__(self, model_path, face_dir, similarity_threshold=0.6):
        self.face_dir = face_dir
        self.threshold = similarity_threshold 
        self.known_embeddings = []
        self.known_user_ids = []
        
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        
        print(f"[FaceRec] Cosine Mode Initialized. Threshold: {self.threshold}")
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir, exist_ok=True)

    def load_selected_users(self, user_ids):
        """선택된 사용자 로드"""
        temp_embeddings = []
        temp_user_ids = []
        
        if not user_ids:
            self.known_embeddings = []
            self.known_user_ids = []
            return
        
        for user_id in user_ids:
            user_path = os.path.join(self.face_dir, user_id)
            emb_file = os.path.join(user_path, "embedding.npy")
            
            if os.path.exists(emb_file):
                try:
                    emb = np.load(emb_file)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    temp_embeddings.append(emb)
                    temp_user_ids.append(user_id)
                except Exception:
                    pass

        self.known_embeddings = temp_embeddings
        self.known_user_ids = temp_user_ids
        print(f"[FaceRec] Loaded Users: {len(self.known_user_ids)}")

    def get_embedding(self, face_img):
        """이미지 -> 정규화된 임베딩 벡터 변환"""
        if face_img is None or face_img.size == 0:
            return None
        try:
            img = cv2.resize(face_img, tuple(self.input_shape))
            img = img.astype(np.float32)
            img = (img - 127.5) / 128.0
            img = np.expand_dims(img, axis=0)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            print(f"[FaceRec] Embedding Error: {e}")
            return None

    def recognize(self, face_crop):
        """코사인 유사도 기반 인식"""
        if not self.known_embeddings:
            return None, 0.0
        
        target_embedding = self.get_embedding(face_crop)
        if target_embedding is None: 
            return None, 0.0

        sims = np.dot(self.known_embeddings, target_embedding)
        
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        
        if best_sim > self.threshold:
            return self.known_user_ids[best_idx], best_sim
        
        return None, 0.0

    def register_user(self, user_id, username, image_path):
        """사용자 등록"""
        try:
            img = cv2.imread(image_path)
            if img is None: return False
            
            embedding = self.get_embedding(img)
            if embedding is None: return False
            
            user_dir = os.path.join(self.face_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            np.save(os.path.join(user_dir, "embedding.npy"), embedding)
            
            with open(os.path.join(user_dir, "metadata.json"), 'w') as f:
                json.dump({"user_id": user_id, "username": username}, f)
            return True
        except Exception: return False