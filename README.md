![Logo](https://github.com/Solrikk/OptiVerse/blob/main/assets/OpenCV%20-%20result/bee.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/README.md">⭐English ⭐</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

-----------------

# OptiVerse

OptiVerse is a powerful video processing application that combines advanced computer vision and machine learning technologies. With OptiVerse, you can automatically extract and transcribe audio, detect faces and poses, recognize objects, as well as add subtitles and annotations to video files.

## Key Features

- **Audio Extraction**: Automatically extracts the audio track from a video file.
- **Audio Transcription**: Converts speech from audio to text using OpenAI's Whisper model.
- **Face Detection**: Identifies and extracts faces from each frame of the video.
- **Pose Recognition**: Determines poses and key body points using MediaPipe.
- **Object Recognition**: Utilizes the YOLOv5 model to detect and classify objects within the frame.
- **Subtitles Addition**: Automatically generates and overlays subtitles on the video based on transcribed text.
- **Annotations and Overlays**: Adds bounding boxes around detected faces and objects, as well as displays poses and additional visual elements.
- **Saving Results**: Exports the processed video with annotations and saves extracted faces and objects in separate directories.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/OptiVerse.git
    cd OptiVerse
    ```

2. **Create and Activate a Virtual Environment (Optional):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    **Note:** Ensure that you have all the necessary libraries installed, including `torch`, `whisper`, `ffmpeg`, `opencv-python`, `mediapipe`, and `yolov5`.

4. **Download the YOLOv5 Model:**
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```

    Place the `yolov5l.pt` model file in the project's root directory or specify the path to the model in the code.

## Usage

1. **Prepare a Video File:**
    Place the video you want to process in the project's root directory and update the file path in the `main()` function:
    ```python
    video_path = 'your_video.mp4'
    ```

2. **Run the Script:**
    ```bash
    python optiverse.py
    ```

3. **Results:**
    - The processed video will be saved as `output_with_faces_and_pose_and_subtitles.mp4`.
    - Extracted faces will be saved in the `extracted_faces` directory.
    - Extracted objects will be saved in the `extracted_object` directory.
    - Subtitles will be saved in the `subtitles.txt` file.

# 🔧 Detailed Feature Description

1. **Audio Extraction (`extract_audio`)**:
The `extract_audio` function uses `ffmpeg` to extract the audio track from a video file and save it in WAV format. This allows you to work with the audio separately for subsequent transcription.

2. **Audio Transcription (`transcribe_audio`)**:
The `transcribe_audio` function applies the Whisper model to convert audio into text. The result includes segments with timestamps and transcribed text, which are used to create subtitles.

3. **Face Detection and Capture (`detect_and_capture_faces`)**:
Using the Haar cascade classifier, the `detect_and_capture_faces` function detects faces in each video frame, highlights them, and saves separate face images.

4. **Pose Recognition (`detect_pose_on_frame`)**:
The `detect_pose_on_frame` function utilizes MediaPipe to determine poses and key body points within the frame. The results are displayed on the video using visual overlays.

5. **Subtitles Addition (`add_subtitles`)**:
The `add_subtitles` function overlays subtitles onto the video based on the current frame's timestamp and the corresponding segments of transcribed text.

6. **Saving Extracted Faces (`save_faces`)**:
The `save_faces` function saves detected faces to the `extracted_faces` directory with unique file names based on the frame number and face index.

7. **Placing Detected Faces on the Frame (`place_detected_faces`)**:
The `place_detected_faces` function places the detected face images in the corner of the video, providing a visual representation of all faces found in the frame.

8. **Loading Reference Objects (`load_reference_objects`)**:
The `load_reference_objects` function loads reference object images from the specified directory for subsequent comparison and recognition.

9. **Object Detection on Frame (`detect_objects_on_frame`)**:
The `detect_objects_on_frame` function applies the YOLOv5 model to detect and classify objects within the frame, returning a list of detected objects with their coordinates.

10. **Placing Reference Objects on the Frame (`place_reference_objects`)**:
The `place_reference_objects` function places images of recognized objects in the corner of the video, similar to how faces are handled.

11. **Video Processing (`process_video`)**:
The main `process_video` function combines all the previous functions for sequential video processing: extracting faces, recognizing poses, adding subtitles, detecting objects, and saving results.

12. **Main Function (`main`)**:
The `main` function manages the video processing workflow, including audio extraction, transcription, subtitle saving, and initiating video processing using MediaPipe for pose recognition.
