![Logo](https://github.com/Solrikk/OptiVerse/blob/main/assets/OpenCV%20-%20result/bee.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/README.md">English</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_CN.md">⭐Chinese⭐</a>
  </h3>
</div>

-----------------

# OptiVerse

OptiVerse 是一款功能强大的视频处理应用程序，结合了先进的计算机视觉和机器学习技术。使用 OptiVerse，您可以自动提取和转录音频，检测人脸和姿势，识别对象，以及为视频文件添加字幕和注释。

## 主要功能

- **音频提取**: 自动从视频文件中提取音轨。
- **音频转录**: 使用 OpenAI 的 Whisper 模型将音频中的语音转换为文本。
- **人脸检测**: 识别并提取视频每一帧中的人脸。
- **姿势识别**: 使用 MediaPipe 确定姿势和关键身体点。
- **对象识别**: 利用 YOLOv5 模型检测并分类帧中的对象。
- **字幕添加**: 根据转录的文本自动生成并叠加字幕到视频中。
- **注释和覆盖**: 在检测到的人脸和对象周围添加边界框，并显示姿势和其他视觉元素。
- **结果保存**: 导出带有注释的处理后视频，并将提取的人脸和对象保存到单独的目录中。

## 安装

1. **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/OptiVerse.git
    cd OptiVerse
    ```

2. **创建并激活虚拟环境（可选）:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # 对于 Linux/Mac
    venv\Scripts\activate     # 对于 Windows
    ```

3. **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **注意:** 确保已安装所有必要的库，包括 `torch`、`whisper`、`ffmpeg`、`opencv-python`、`mediapipe` 和 `yolov5`。

4. **下载 YOLOv5 模型:**
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```
    
    将 `yolov5l.pt` 模型文件放置在项目的根目录，或在代码中指定模型的路径。

## 使用方法

1. **准备视频文件:**
    将您想要处理的视频放置在项目的根目录，并在 `main()` 函数中更新文件路径：
    ```python
    video_path = 'your_video.mp4'
    ```

2. **运行脚本:**
    ```bash
    python optiverse.py
    ```

3. **结果:**
    - 处理后的视频将保存为 `output_with_faces_and_pose_and_subtitles.mp4`。
    - 提取的人脸将保存在 `extracted_faces` 目录中。
    - 提取的对象将保存在 `extracted_object` 目录中。
    - 字幕将保存在 `subtitles.txt` 文件中。

# 🔧 详细功能描述

OptiVerse 利用一系列先进的技术提供强大的视频处理能力。以下是每个功能的技术概述，重点介绍了关键组件、方法论和相关公式。

## 1. **音频提取 (`extract_audio`)**

`extract_audio` 函数使用 `ffmpeg` 库从给定的视频文件中提取音频流。通过将音频转换为 WAV 格式，确保了与各种转录模型的兼容性。此函数处理临时文件的创建，并包含错误处理，以有效管理提取过程中的任何问题。

## 2. **音频转录 (`transcribe_audio`)**

利用 OpenAI 的 Whisper 模型，`transcribe_audio` 函数将提取的音频转换为文本。Whisper 强大的转录能力提供了准确且带有时间戳的段落，对于同步字幕与视频至关重要。转录过程可以通过 **Transformer 架构** 的数学描述来解释，其中输入的音频特征被转换为一系列文本标记。

**Transformer 方程:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

其中:
- \( Q \) = 查询矩阵（Query matrix）
- \( K \) = 键矩阵（Key matrix）
- \( V \) = 值矩阵（Value matrix）
- \( d_k \) = 键向量的维度

这种注意力机制使模型能够在生成转录文本的每一部分时，对输入音频的不同部分赋予相关性的权重。

## 3. **人脸检测与捕获 (`detect_and_capture_faces`)**

该功能使用 OpenCV 的 Haar 级联分类器从视频的每一帧中识别和提取人脸。通过将帧处理为灰度图像，函数增强了检测的准确性和性能。检测到的人脸用边界框突出显示，并作为单独的图像文件保存，以便将来的分析或如人脸识别等应用。

**Haar 级联分类器:**

分类器使用 Haar 类似的特征，通过在多个尺度和位置扫描图像来检测对象（人脸）。

**积分图计算:**

\[
\text{Integral Image}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
\]

其中 \( I(i, j) \) 是位置 \( (i, j) \) 的像素强度。

这允许快速计算 Haar 特征，从而实现高效的人脸检测。

## 4. **姿势识别 (`detect_pose_on_frame`)**

利用 MediaPipe 的 Pose 解决方案，`detect_pose_on_frame` 函数在每个视频帧中识别关键身体地标。这使得通过在视频上叠加骨架连接来可视化人类姿势成为可能，特别适用于运动分析、体育训练和动画等应用。

**关键点检测:**

MediaPipe 使用包括以下内容的多阶段管道:
1. **地标检测:** 识别关键身体点。
2. **姿势估计:** 确定地标之间的空间关系。

**关节角度计算示例方程:**

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)
\]

其中 \( \mathbf{u} \) 和 \( \mathbf{v} \) 是代表四肢的向量，\( \theta \) 是它们之间的角度。

## 5. **字幕添加 (`add_subtitles`)**

`add_subtitles` 函数通过叠加同步字幕将转录的文本集成到视频中。它将当前播放时间与相应的转录段落匹配，确保字幕准确反映所说的音频内容。这提高了可访问性并提供了更好的观看体验。

**时间戳匹配:**

\[
\text{Subtitle Time Window} = [t_{\text{start}}, t_{\text{end}}]
\]

当当前视频时间 \( t \) 满足以下条件时，字幕会显示:

\[
t_{\text{start}} \leq t \leq t_{\text{end}}
\]

## 6. **保存提取的人脸 (`save_faces`)**

检测到的人脸使用包含帧编号和人脸索引的唯一文件名系统地保存到 `extracted_faces` 目录中。这种有组织的存储便于后续处理或分析的人脸图像的检索和管理。

**文件名约定:**

\[
\text{filename} = \text{"frame\_"} + \text{frame\_count} + \text{"\_face\_"} + \text{face\_index} + \text{".png"}
\]

## 7. **在帧上放置检测到的人脸 (`place_detected_faces`)**

为了提供检测到的人脸的视觉摘要，`place_detected_faces` 函数将提取的人脸图像叠加到视频帧的指定区域。通过适当调整大小和定位这些图像，确保视频的主要内容不受阻碍，同时显示相关的人脸信息。

**覆盖位置:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (\text{width} - 120, 10 + 110 \times \text{idx})
\]

其中:
- \( \text{idx} \) = 检测到的人脸的索引

## 8. **加载参考对象 (`load_reference_objects`)**

此功能从指定目录加载对象的参考图像。这些参考图像作为对象识别的基准，使应用程序能够比较并识别视频帧中检测到的对象。这对于对象跟踪和分类等任务至关重要。

**图像加载:**

\[
\text{ref\_objects}[ \text{class\_name.lower()} ] = \text{cv2.imread}(\text{file\_path})
\]

## 9. **在帧上检测对象 (`detect_objects_on_frame`)**

使用 YOLOv5 模型，`detect_objects_on_frame` 函数在每个视频帧中检测和分类对象。YOLOv5 的实时对象检测能力使其能够高效地识别多个对象，并提供其类别名称和边界框坐标，以实现准确的位置定位和分类。

**YOLOv5 检测:**

YOLOv5 将图像划分为网格，并为每个网格单元预测边界框和类别概率。

**边界框预测:**

\[
(x, y, w, h, \text{confidence}, \text{class\_scores})
\]

其中:
- \( x, y \) = 中心坐标
- \( w, h \) = 宽度和高度
- \( \text{confidence} \) = 物体置信度分数
- \( \text{class\_scores} \) = 每个类别的概率

## 10. **在帧上放置参考对象 (`place_reference_objects`)**

类似于人脸放置，此功能将识别的对象图像叠加到视频帧上。通过在特定区域定位这些参考对象图像，它提供了在整个视频中检测到的对象的清晰视觉表示，增强了处理内容的信息价值。

**覆盖位置:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (20, 10 + 110 \times \text{idx})
\]

其中:
- \( \text{idx} \) = 检测到的对象的索引

## 11. **视频处理 (`process_video`)**

`process_video` 函数组织了全面的视频处理工作流程。它依次对输入视频的每一帧应用人脸检测、姿势识别、字幕添加和对象检测。处理后的帧被编译成最终的输出视频，其中包含所有注释和覆盖，确保连贯和丰富的观看体验。

**工作流程步骤:**
1. **帧提取:** 使用 `cv2.VideoCapture` 从输入视频中读取帧。
2. **功能应用:** 对每一帧应用检测和注释功能。
3. **帧编译:** 使用 `cv2.VideoWriter` 将注释过的帧写入输出视频文件。

**帧处理循环:**

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

## 12. **主函数 (`main`)**

作为入口点，`main` 函数管理 OptiVerse 应用程序的整体执行。它启动音频提取和转录，处理字幕生成，并触发视频处理管道。此外，它还集成了日志记录机制，以跟踪处理阶段并处理可能出现的任何异常，确保顺利和可靠的操作。

**执行流程:**
1. **音频提取:** 调用 `extract_audio` 以检索音频轨道。
2. **转录:** 将提取的音频传递给 `transcribe_audio` 进行文本转换。
3. **字幕保存:** 将转录的段落写入 `subtitles.txt` 文件。
4. **姿势初始化:** 初始化 MediaPipe 的 Pose 解决方案。
5. **视频处理:** 使用视频路径、转录段落和姿势对象调用 `process_video`。
6. **日志记录:** 记录处理过程中遇到的状态和任何错误。
