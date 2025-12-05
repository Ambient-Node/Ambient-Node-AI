FROM python:3.10.12-slim


WORKDIR /app


# 런타임 라이브러리
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libavformat59 \
    libavcodec59 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*


# 1) 라즈베리 파이와 동일한 핵심 조합을 먼저 강제 설치
COPY requirements.txt .


RUN pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple \
      numpy==1.26.4 \
    && pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple \
      --no-deps opencv-python==4.12.0.88 opencv-contrib-python==4.12.0.88 \
    && pip install --no-cache-dir --extra-index-url https://www.piwheels.org/simple \
      -r requirements.txt --no-deps


# 2) 코드 & 모델 복사
COPY facenet.tflite /app/facenet.tflite
COPY *.py /app/


RUN mkdir -p /var/lib/ambient-node/users /var/lib/ambient-node/captures


ENV PYTHONUNBUFFERED=1
ENV MQTT_BROKER=localhost
ENV MQTT_PORT=1883
ENV TFLITE_MODEL_PATH=/app/facenet.tflite


CMD ["python3", "main.py"]
