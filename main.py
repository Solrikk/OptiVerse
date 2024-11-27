import torch
import whisper
import tempfile
import os
import ffmpeg
import shutil
import cv2
import numpy as np
import sys

sys.path.append('yolov5')
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

whisper_model = whisper.load_model("base")
device = select_device('cpu')
yolo_model = DetectMultiBackend('yolov5l.pt', device=device)
imgsz = [640, 640]

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

def detect_and_annotate(frame, model, device, imgsz):
    img = cv2.resize(frame, (imgsz[0], imgsz[1]))
    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img / 255.0
    img = torch.from_numpy(img).float().to(device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.4, classes=None, agnostic=False, max_det=1000)

    annotator = Annotator(frame, line_width=3, font_size=1, example=str(model.names))
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape[:2]).round()
            for *xyxy, conf, cls in reversed(det):
                label = f"{model.names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
    return annotator.result()

def process_video(video_path: str, segments):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = 'output_with_subtitles_and_objects.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    subtitles = []
    for segment in segments:
        start_frame = int(segment['start'] * fps)
        end_frame = int(segment['end'] * fps)
        text = segment['text']
        subtitles.append((start_frame, end_frame, text))
    
    frame_num = 0
    subtitle_index = 0
    current_subtitle = ''
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = detect_and_annotate(frame, yolo_model, device, imgsz)
        
        if subtitle_index < len(subtitles):
            start_frame_sub, end_frame_sub, text_sub = subtitles[subtitle_index]
            if frame_num >= start_frame_sub and frame_num <= end_frame_sub:
                current_subtitle = text_sub
            elif frame_num > end_frame_sub:
                subtitle_index += 1
                current_subtitle = ''
                
        if current_subtitle:
            cv2.putText(frame, current_subtitle, (50, height - 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        frame_num += 1
        
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