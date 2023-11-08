FROM python:3.10.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY config/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
