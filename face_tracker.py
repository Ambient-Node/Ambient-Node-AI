"""얼굴 추적 관리 (단순 거리 계산 방식 복구)"""

import threading

class FaceTracker:
    def __init__(self, max_distance=150, lost_timeout=0.5, enable_display=True):
        self.tracked_faces = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.lost_timeout = lost_timeout
        self.lock = threading.Lock()
            
    def reset(self):
        with self.lock:
            self.tracked_faces.clear()
            self.next_id = 0

    def update(self, detected_positions, current_time):
        """이전 방식: 가장 가까운 트래커 찾기"""
        with self.lock:
            updated_ids = set()
            
            for pos in detected_positions:
                center = pos['center']
                # 단순 거리 계산 (Greedy)
                closest_id = self._find_closest(center)
                
                if closest_id is not None:
                    self.tracked_faces[closest_id].update({
                        'center': center,
                        'bbox': pos['bbox'],
                        'last_seen': current_time,
                    })
                    updated_ids.add(closest_id)
                else:
                    self.tracked_faces[self.next_id] = {
                        'user_id': None,
                        'center': center,
                        'bbox': pos['bbox'],
                        'last_seen': current_time,
                        'last_identified': 0.0,
                    }
                    updated_ids.add(self.next_id)
                    self.next_id += 1
            
            lost_faces = self._remove_expired(current_time, timeout=self.lost_timeout)
            return updated_ids, lost_faces

    def _find_closest(self, center):
        """단순 유클리드 거리 계산"""
        min_dist = float('inf')
        closest_id = None
        
        for fid, finfo in self.tracked_faces.items():
            old_center = finfo['center']
            dist = ((center[0] - old_center[0]) ** 2 +
                   (center[1] - old_center[1]) ** 2) ** 0.5
            if dist < min_dist and dist < self.max_distance:
                min_dist = dist
                closest_id = fid
        
        return closest_id

    def _remove_expired(self, current_time, timeout):
        expired = [fid for fid, finfo in self.tracked_faces.items()
                   if current_time - finfo['last_seen'] > timeout]
        
        lost_faces = []
        for fid in expired:
            uid = self.tracked_faces[fid].get('user_id')
            if uid: lost_faces.append({'user_id': uid, 'duration': 0})
            del self.tracked_faces[fid]
        return lost_faces

    def identify_faces(self, recognizer, frame, current_time, interval, force_all=False):
        with self.lock:
            newly_identified = []
            
            for fid, finfo in self.tracked_faces.items():
                if not force_all and current_time - finfo['last_identified'] < interval:
                    continue
                
                x1, y1, x2, y2 = finfo['bbox']
                # FHD 프레임 범위 체크
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue
                
                user_id, confidence = recognizer.recognize(face_crop)
                
                if user_id:
                    if finfo.get('user_id') == user_id:
                        confidence = min(0.95, confidence + 0.05)
                    
                    finfo['user_id'] = user_id
                    finfo['confidence'] = confidence
                    finfo['last_identified'] = current_time
                    if 'first_seen' not in finfo: finfo['first_seen'] = current_time
                    
                    newly_identified.append((fid, user_id, confidence))
            
            return newly_identified

    def get_selected_faces(self, selected_user_ids):
        with self.lock:
            return [
                {**finfo, 'face_id': fid}
                for fid, finfo in self.tracked_faces.items()
                if finfo.get('user_id') in selected_user_ids
            ]