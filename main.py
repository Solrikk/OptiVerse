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
import logging
from typing import List, Dict, Tuple

sys.path.append('yolov5')
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

whisper_model = whisper.load_model("base")
device = select_device('cpu')
yolo_model = DetectMultiBackend('yolov5l.pt', device=device)
imgsz = [640, 640]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_audio(video_file_path: str) -> str:
    audio_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_output.close()
    try:
        ffmpeg.input(video_file_path).output(audio_output.name, format='wav').run(overwrite_output=True)
        return audio_output.name
    except Exception as e:
        if os.path.exists(audio_output.name):
            os.unlink(audio_output.name)
        raise e


def transcribe_audio(audio_file_path: str) -> List[Dict]:
    result = whisper_model.transcribe(audio_file_path)
    segments = result.get('segments', [])
    return segments


def detect_and_capture_faces(frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    face_images = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_images.append(face_img)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame, face_images


def detect_pose_on_frame(frame: np.ndarray, pose) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    return frame


def add_subtitles(frame: np.ndarray, segments: List[Dict], current_time_ms: float) -> None:
    subtitle_text = None
    for segment in segments:
        start_time = segment['start'] * 1000
        end_time = segment['end'] * 1000
        text = segment['text']
        if start_time <= current_time_ms <= end_time:
            subtitle_text = text.strip()
            break
    if subtitle_text:
        text = subtitle_text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = 50
        y = frame.shape[0] - 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - text_h - 10), (x + text_w + 10, y + 10), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def save_faces(face_images: List[np.ndarray], output_dir: str, frame_count: int) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i, face in enumerate(face_images):
        face_filename = os.path.join(output_dir, f"frame_{frame_count}_face_{i}.png")
        cv2.imwrite(face_filename, face)


def place_detected_faces(frame: np.ndarray, face_images: List[np.ndarray], width: int, height: int) -> None:
    for idx, face in enumerate(face_images):
        face = cv2.resize(face, (100, 100))
        x_offset = width - 120
        y_offset = 10 + (110 * idx)
        if y_offset + 100 <= height:
            frame[y_offset:y_offset + 100, x_offset:x_offset + 100] = face


def load_reference_objects(object_dir: str) -> Dict[str, np.ndarray]:
    ref_objects = {}
    if os.path.exists(object_dir):
        for fname in os.listdir(object_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = os.path.splitext(fname)[0]
                obj_img = cv2.imread(os.path.join(object_dir, fname))
                if obj_img is not None:
                    ref_objects[class_name.lower()] = obj_img
    return ref_objects


def detect_objects_on_frame(frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    img = [frame]
    pred = yolo_model(img, size=imgsz)
    pred = non_max_suppression(pred)[0]
    detections = []
    if pred is not None and len(pred):
        for *box, conf, cls in pred:
            x1, y1, x2, y2 = map(int, box)
            cls_name = yolo_model.names[int(cls)].lower()
            detections.append((cls_name, (x1, y1, x2, y2)))
    return detections


def place_reference_objects(frame: np.ndarray, objects_images: List[np.ndarray], width: int, height: int) -> None:
    for idx, obj_img in enumerate(objects_images):
        obj_img = cv2.resize(obj_img, (100, 100))
        x_offset = 20
        y_offset = 10 + (110 * idx)
        if y_offset + 100 <= height:
            frame[y_offset:y_offset + 100, x_offset:x_offset + 100] = obj_img


def process_video(video_path: str, segments: List[Dict], pose) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = 'output_with_faces_and_pose_and_subtitles.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_output_dir = 'extracted_faces'
    object_output_dir = 'extracted_object'
    reference_objects = load_reference_objects(object_output_dir)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame, face_images = detect_and_capture_faces(frame)
            frame = detect_pose_on_frame(frame, pose)

            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            add_subtitles(frame, segments, current_time_ms)

            if face_images:
                save_faces(face_images, face_output_dir, frame_count)

            place_detected_faces(frame, face_images, width, height)

            objects_found = detect_objects_on_frame(frame)
            matched_objects_images = []
            for cls_name, (x1, y1, x2, y2) in objects_found:
                color = (255, 0, 0)
                if cls_name in reference_objects:
                    color = (0, 255, 0)
                    matched_objects_images.append(reference_objects[cls_name])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if matched_objects_images:
                place_reference_objects(frame, matched_objects_images, width, height)

            out.write(frame)
    finally:
        cap.release()
        out.release()

    return output_path


def main():
    video_path = 'video.mp4'
    try:
        audio_path = extract_audio(video_path)
        segments = transcribe_audio(audio_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        with mp_pose.Pose(min_detection_confidence=0.5) as pose:
            output_path = process_video(video_path, segments, pose)
        logging.info(f"Обработанное видео сохранено в {output_path}")
    except Exception as e:
        logging.error(f'Failed to process video: {str(e)}')


if __name__ == "__main__":
    main()
