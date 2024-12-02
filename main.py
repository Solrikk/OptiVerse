import torch
import whisper
import tempfile
import os
import ffmpeg
import shutil
import cv2
import numpy as np
import sys
import mediapipe as mp

sys.path.append('yolov5')
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

whisper_model = whisper.load_model("base")
device = select_device('cpu')
yolo_model = DetectMultiBackend('yolov5l.pt', device=device)
imgsz = [640, 640]

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_audio(video_file_path: str) -> str:
  audio_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
  audio_output.close()
  try:
    ffmpeg.input(video_file_path).output(
        audio_output.name, format='wav').run(overwrite_output=True)
    return audio_output.name
  except Exception as e:
    os.unlink(audio_output.name)
    raise e


def transcribe_audio(audio_file_path: str):
  result = whisper_model.transcribe(audio_file_path)
  segments = result['segments']
  return segments


def detect_and_capture_faces(frame):
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                       'haarcascade_frontalface_default.xml')
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray_frame,
                                        scaleFactor=1.1,
                                        minNeighbors=5)

  face_images = []

  for (x, y, w, h) in faces:
    face_img = frame[y:y + h, x:x + w]
    face_images.append(face_img)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

  return frame, face_images


def detect_pose(frame):
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  with mp_pose.Pose(min_detection_confidence=0.5) as pose:
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
      mp_drawing.draw_landmarks(
          frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
          mp_drawing.DrawingSpec(color=(245, 117, 66),
                                 thickness=2,
                                 circle_radius=2),
          mp_drawing.DrawingSpec(color=(245, 66, 230),
                                 thickness=2,
                                 circle_radius=2))

  return frame


def process_video(video_path: str, segments):
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  output_path = 'output_with_faces_and_pose.mp4'
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame, face_images = detect_and_capture_faces(frame)
    frame = detect_pose(frame)

    for idx, face in enumerate(face_images):
      face = cv2.resize(face, (100, 100))
      x_offset = width - 120
      y_offset = 10 + (110 * idx)
      if y_offset + 100 <= height:
        frame[y_offset:y_offset + 100, x_offset:x_offset + 100] = face

    out.write(frame)

  cap.release()
  out.release()
  return output_path


def main():
  video_path = 'your_video.mp4'
  audio_path = extract_audio(video_path)
  segments = transcribe_audio(audio_path)
  os.unlink(audio_path)
  output_path = process_video(video_path, segments)
  print(f"Обработанное видео сохранено в {output_path}")


if __name__ == "__main__":
  main()
