FROM python:3.10-slim

WORKDIR /app
COPY . /app

# For OpenCV & ffmpeg & soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

ENV PORT=10000
EXPOSE 10000

CMD ["python", "app.py"]
