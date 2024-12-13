![Logo](https://github.com/Solrikk/dectralv1/blob/main/assets/photo/jj-ying-0MT.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/dectralv1/blob/main/README.md">⭐English⭐</a> |
    <a href="https://github.com/Solrikk/dectralv1/blob/main/docs/readme/README_RU.md">Русский</a> |
    <a href="https://github.com/Solrikk/dectralv1/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/dectralv1/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/dectralv1/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/dectralv1/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

-----------------

# dectralv1

dectralv1 is a powerful video processing application that combines advanced computer vision and machine learning technologies. With dectralv1, you can automatically extract and transcribe audio, detect faces and poses, recognize objects, as well as add subtitles and annotations to video files.

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
    git clone https://github.com/your-username/dectralv1.git
    cd dectralv1
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
    python dectralv1.py
    ```

3. **Results:**
    - The processed video will be saved as `output_with_faces_and_pose_and_subtitles.mp4`.
    - Extracted faces will be saved in the `extracted_faces` directory.
    - Extracted objects will be saved in the `extracted_object` directory.
    - Subtitles will be saved in the `subtitles.txt` file.

# 🔧 Detailed Feature Description

dectralv1 leverages a suite of advanced technologies to deliver robust video processing capabilities. Below is a technical overview of each feature, highlighting key components, methodologies, and relevant formulas employed.

## 1. **Audio Extraction (`extract_audio`)**

The `extract_audio` function employs the `ffmpeg` library to extract the audio stream from a given video file. By converting the audio to WAV format, it ensures compatibility with various transcription models. This function handles the creation of temporary files and includes error handling to manage any issues during the extraction process efficiently.

## 2. **Audio Transcription (`transcribe_audio`)**

Utilizing OpenAI's Whisper model, the `transcribe_audio` function converts the extracted audio into text. Whisper's robust transcription capabilities provide accurate and timestamped segments, essential for synchronizing subtitles with the video. The transcription process can be mathematically described using the **Transformer Architecture**, where the input audio features are transformed into a sequence of text tokens.

**Transformer Equation:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Where:
- \( Q \) = Query matrix
- \( K \) = Key matrix
- \( V \) = Value matrix
- \( d_k \) = Dimension of the key vectors

This attention mechanism allows the model to weigh the relevance of different parts of the input audio when generating each part of the transcribed text.

## 3. **Face Detection and Capture (`detect_and_capture_faces`)**

This feature uses OpenCV's Haar cascade classifier to identify and extract faces from each frame of the video. By processing frames in grayscale, the function enhances detection accuracy and performance. Detected faces are highlighted with bounding boxes and saved as separate image files for potential future analysis or applications like face recognition.

**Haar Cascade Classifier:**

The classifier uses Haar-like features to detect objects (faces) by scanning the image at multiple scales and positions.

**Integral Image Calculation:**

\[
\text{Integral Image}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
\]

Where \( I(i, j) \) is the pixel intensity at position \( (i, j) \).

This allows for rapid computation of Haar features, enabling efficient face detection.

## 4. **Pose Recognition (`detect_pose_on_frame`)**

Leveraging MediaPipe's Pose solution, the `detect_pose_on_frame` function identifies key body landmarks within each video frame. This enables the visualization of human poses by overlaying skeletal connections on the video, which is particularly useful for applications in motion analysis, sports training, and animation.

**Keypoint Detection:**

MediaPipe uses a multi-stage pipeline that includes:
1. **Landmark Detection:** Identifies key body points.
2. **Pose Estimation:** Determines the spatial relationships between landmarks.

**Example Equation for Joint Angle Calculation:**

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)
\]

Where \( \mathbf{u} \) and \( \mathbf{v} \) are vectors representing limbs, and \( \theta \) is the angle between them.

## 5. **Subtitles Addition (`add_subtitles`)**

The `add_subtitles` function integrates the transcribed text into the video by overlaying synchronized subtitles. It matches the current playback time with the corresponding transcription segments, ensuring that the subtitles accurately reflect the spoken audio. This enhances accessibility and provides a better viewing experience.

**Timestamp Matching:**

\[
\text{Subtitle Time Window} = [t_{\text{start}}, t_{\text{end}}]
\]

Subtitles are displayed when the current video time \( t \) satisfies:

\[
t_{\text{start}} \leq t \leq t_{\text{end}}
\]

## 6. **Saving Extracted Faces (`save_faces`)**

Detected faces are systematically saved to the `extracted_faces` directory using unique filenames that incorporate the frame number and face index. This organized storage facilitates easy retrieval and management of face images for further processing or analysis.

**Filename Convention:**

\[
\text{filename} = \text{"frame\_"} + \text{frame\_count} + \text{"\_face\_"} + \text{face\_index} + \text{".png"}
\]

## 7. **Placing Detected Faces on the Frame (`place_detected_faces`)**

To provide a visual summary of detected faces, the `place_detected_faces` function overlays the extracted face images onto a designated area of the video frame. By resizing and positioning these images appropriately, it ensures that the main content of the video remains unobstructed while still displaying relevant face information.

**Overlay Positioning:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (\text{width} - 120, 10 + 110 \times \text{idx})
\]

Where:
- \( \text{idx} \) = Index of the detected face

## 8. **Loading Reference Objects (`load_reference_objects`)**

This function loads reference images of objects from a specified directory. These reference images serve as a baseline for object recognition, allowing the application to compare and identify objects detected in the video frames. This is essential for tasks such as object tracking and classification.

**Image Loading:**

\[
\text{ref\_objects}[ \text{class\_name.lower()} ] = \text{cv2.imread}(\text{file\_path})
\]

## 9. **Object Detection on Frame (`detect_objects_on_frame`)**

Employing the YOLOv5 model, the `detect_objects_on_frame` function detects and classifies objects within each video frame. YOLOv5's real-time object detection capabilities enable efficient identification of multiple objects, providing their class names and bounding box coordinates for accurate localization and categorization.

**YOLOv5 Detection:**

YOLOv5 divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell.

**Bounding Box Prediction:**

\[
(x, y, w, h, \text{confidence}, \text{class\_scores})
\]

Where:
- \( x, y \) = Center coordinates
- \( w, h \) = Width and height
- \( \text{confidence} \) = Objectness score
- \( \text{class\_scores} \) = Probabilities for each class

## 10. **Placing Reference Objects on the Frame (`place_reference_objects`)**

Similar to face placement, this function overlays images of recognized objects onto the video frame. By positioning these reference object images in a specific area, it offers a clear visual representation of the objects detected throughout the video, enhancing the informational value of the processed content.

**Overlay Positioning:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (20, 10 + 110 \times \text{idx})
\]

Where:
- \( \text{idx} \) = Index of the detected object

## 11. **Video Processing (`process_video`)**

The `process_video` function orchestrates the comprehensive video processing workflow. It sequentially applies face detection, pose recognition, subtitle addition, and object detection to each frame of the input video. The processed frames are then compiled into a final output video that includes all annotations and overlays, ensuring a cohesive and enriched viewing experience.

**Workflow Steps:**
1. **Frame Extraction:** Read frames from the input video using `cv2.VideoCapture`.
2. **Feature Application:** Apply detection and annotation functions to each frame.
3. **Frame Compilation:** Write the annotated frames to an output video file using `cv2.VideoWriter`.

**Frame Processing Loop:**

\[
\text{for each frame in video:}
\]
\[
\quad \text{detect\_and\_capture\_faces(frame)}
\]
\[
\quad \text{detect\_pose\_on\_frame(frame, pose)}
\]
\[
\quad \text{add\_subtitles(frame, segments, current\_time)}
\]
\[
\quad \text{detect\_objects\_on\_frame(frame)}
\]
\[
\quad \text{save\_faces(face\_images, output\_dir, frame\_count)}
\]
\[
\quad \text{place\_detected\_faces(frame, face\_images, width, height)}
\]
\[
\quad \text{place\_reference\_objects(frame, matched\_objects, width, height)}
\]
\[
\quad \text{write\_frame\_to\_output(frame)}
\]

## 12. **Main Function (`main`)**

Serving as the entry point, the `main` function manages the overall execution of the dectralv1 application. It initiates audio extraction and transcription, handles subtitle generation, and triggers the video processing pipeline. Additionally, it incorporates logging mechanisms to track the processing stages and handle any exceptions that may arise, ensuring a smooth and reliable operation.

**Execution Flow:**
1. **Audio Extraction:** Call `extract_audio` to retrieve the audio track.
2. **Transcription:** Pass the extracted audio to `transcribe_audio` for text conversion.
3. **Subtitle Saving:** Write the transcription segments to a `subtitles.txt` file.
4. **Pose Initialization:** Initialize MediaPipe's Pose solution.
5. **Video Processing:** Invoke `process_video` with the video path, transcription segments, and pose object.
6. **Logging:** Record the status and any errors encountered during processing.




