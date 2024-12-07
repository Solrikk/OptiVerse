![Logo](https://github.com/Solrikk/OptiVerse/blob/main/assets/OpenCV%20-%20result/bee.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/README.md">English</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_KR.md">⭐Korean⭐</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

-----------------

# OptiVerse

OptiVerse는 첨단 컴퓨터 비전 및 기계 학습 기술을 결합한 강력한 비디오 처리 애플리케이션입니다. OptiVerse를 사용하면 비디오 파일에서 오디오를 자동으로 추출 및 전사하고, 얼굴과 포즈를 감지하며, 객체를 인식하고, 비디오에 자막과 주석을 추가할 수 있습니다.

## 주요 기능

- **오디오 추출**: 비디오 파일에서 오디오 트랙을 자동으로 추출합니다.
- **오디오 전사**: OpenAI의 Whisper 모델을 사용하여 오디오의 음성을 텍스트로 변환합니다.
- **얼굴 감지**: 비디오의 각 프레임에서 얼굴을 식별하고 추출합니다.
- **포즈 인식**: MediaPipe를 사용하여 포즈 및 주요 신체 지점을 결정합니다.
- **객체 인식**: YOLOv5 모델을 활용하여 프레임 내의 객체를 감지하고 분류합니다.
- **자막 추가**: 전사된 텍스트를 기반으로 비디오에 자동으로 자막을 생성하고 오버레이합니다.
- **주석 및 오버레이**: 감지된 얼굴과 객체 주위에 경계 상자를 추가하고, 포즈 및 추가 시각 요소를 표시합니다.
- **결과 저장**: 주석이 포함된 처리된 비디오를 내보내고 추출된 얼굴과 객체를 별도의 디렉토리에 저장합니다.

## 설치

1. **리포지토리 클론하기:**
    ```bash
    git clone https://github.com/your-username/OptiVerse.git
    cd OptiVerse
    ```

2. **가상 환경 생성 및 활성화 (선택 사항):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac용
    venv\Scripts\activate     # Windows용
    ```

3. **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **참고:** `torch`, `whisper`, `ffmpeg`, `opencv-python`, `mediapipe`, 및 `yolov5`를 포함한 모든 필요한 라이브러리가 설치되었는지 확인하십시오.

4. **YOLOv5 모델 다운로드:**
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```
    
    `yolov5l.pt` 모델 파일을 프로젝트의 루트 디렉토리에 배치하거나 코드에서 모델 경로를 지정하십시오.

## 사용법

1. **비디오 파일 준비:**
    처리하려는 비디오를 프로젝트의 루트 디렉토리에 배치하고 `main()` 함수에서 파일 경로를 업데이트하십시오.
    ```python
    video_path = 'your_video.mp4'
    ```

2. **스크립트 실행:**
    ```bash
    python optiverse.py
    ```

3. **결과:**
    - 처리된 비디오는 `output_with_faces_and_pose_and_subtitles.mp4`로 저장됩니다.
    - 추출된 얼굴은 `extracted_faces` 디렉토리에 저장됩니다.
    - 추출된 객체는 `extracted_object` 디렉토리에 저장됩니다.
    - 자막은 `subtitles.txt` 파일에 저장됩니다.

# 🔧 상세 기능 설명

OptiVerse는 견고한 비디오 처리 기능을 제공하기 위해 다양한 첨단 기술을 통합하고 있습니다. 아래는 각 기능의 기술적 개요로, 주요 구성 요소, 방법론 및 관련 수식을 강조합니다.

## 1. **오디오 추출 (`extract_audio`)**

`extract_audio` 함수는 `ffmpeg` 라이브러리를 사용하여 지정된 비디오 파일에서 오디오 스트림을 추출합니다. 오디오를 WAV 형식으로 변환함으로써 다양한 전사 모델과의 호환성을 보장합니다. 이 함수는 임시 파일 생성과 추출 과정 중 발생할 수 있는 문제를 효율적으로 관리하기 위한 오류 처리도 포함하고 있습니다.

## 2. **오디오 전사 (`transcribe_audio`)**

OpenAI의 Whisper 모델을 활용하여 `transcribe_audio` 함수는 추출된 오디오를 텍스트로 변환합니다. Whisper의 강력한 전사 기능은 비디오와 자막을 동기화하는 데 필수적인 정확하고 타임스탬프가 있는 세그먼트를 제공합니다. 전사 과정은 **Transformer 아키텍처**를 사용하여 입력 오디오 특징을 텍스트 토큰의 시퀀스로 변환하는 수학적으로 설명할 수 있습니다.

**트랜스포머 방정식:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

여기서:
- \( Q \) = 쿼리 행렬 (Query matrix)
- \( K \) = 키 행렬 (Key matrix)
- \( V \) = 값 행렬 (Value matrix)
- \( d_k \) = 키 벡터의 차원

이 어텐션 메커니즘은 모델이 전사 텍스트의 각 부분을 생성할 때 입력 오디오의 다양한 부분의 관련성을 가중치로 평가할 수 있게 합니다.

## 3. **얼굴 감지 및 캡처 (`detect_and_capture_faces`)**

이 기능은 OpenCV의 Haar 캐스케이드 분류기를 사용하여 비디오의 각 프레임에서 얼굴을 식별하고 추출합니다. 프레임을 그레이스케일로 처리함으로써 감지 정확도와 성능을 향상시킵니다. 감지된 얼굴은 경계 상자로 강조 표시되고, 향후 분석이나 얼굴 인식과 같은 애플리케이션을 위해 별도의 이미지 파일로 저장됩니다.

**Haar 캐스케이드 분류기:**

분류기는 Haar 유사 특징을 사용하여 여러 스케일과 위치에서 이미지를 스캔함으로써 객체(얼굴)를 감지합니다.

**적분 이미지 계산:**

\[
\text{Integral Image}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
\]

여기서 \( I(i, j) \)는 위치 \( (i, j) \)에서의 픽셀 강도입니다.

이는 Haar 특징의 신속한 계산을 가능하게 하여 효율적인 얼굴 감지를 가능하게 합니다.

## 4. **포즈 인식 (`detect_pose_on_frame`)**

MediaPipe의 Pose 솔루션을 활용하여 `detect_pose_on_frame` 함수는 각 비디오 프레임 내에서 주요 신체 랜드마크를 식별합니다. 이를 통해 비디오에 골격 연결을 오버레이하여 인간의 포즈를 시각화할 수 있으며, 이는 특히 동작 분석, 스포츠 트레이닝 및 애니메이션과 같은 애플리케이션에 유용합니다.

**키포인트 감지:**

MediaPipe는 다음을 포함하는 다단계 파이프라인을 사용합니다:
1. **랜드마크 감지:** 주요 신체 지점을 식별합니다.
2. **포즈 추정:** 랜드마크 간의 공간적 관계를 결정합니다.

**관절 각도 계산 예제 방정식:**

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)
\]

여기서 \( \mathbf{u} \)와 \( \mathbf{v} \)는 사지를 나타내는 벡터이며, \( \theta \)는 그 사이의 각도입니다.

## 5. **자막 추가 (`add_subtitles`)**

`add_subtitles` 함수는 전사된 텍스트를 비디오에 통합하여 동기화된 자막을 오버레이합니다. 현재 재생 시간을 해당 전사 세그먼트와 일치시켜 자막이 음성으로 말해진 내용을 정확하게 반영하도록 합니다. 이는 접근성을 향상시키고 더 나은 시청 경험을 제공합니다.

**타임스탬프 매칭:**

\[
\text{Subtitle Time Window} = [t_{\text{start}}, t_{\text{end}}]
\]

자막은 현재 비디오 시간 \( t \)가 다음 조건을 만족할 때 표시됩니다:

\[
t_{\text{start}} \leq t \leq t_{\text{end}}
\]

## 6. **추출된 얼굴 저장 (`save_faces`)**

감지된 얼굴은 프레임 번호와 얼굴 인덱스를 포함하는 고유한 파일 이름을 사용하여 `extracted_faces` 디렉토리에 체계적으로 저장됩니다. 이 조직적인 저장은 추가 처리나 분석을 위한 얼굴 이미지의 손쉬운 검색 및 관리를 용이하게 합니다.

**파일 이름 규칙:**

\[
\text{filename} = \text{"frame\_"} + \text{frame\_count} + \text{"\_face\_"} + \text{face\_index} + \text{".png"}
\]

## 7. **프레임에 감지된 얼굴 배치 (`place_detected_faces`)**

감지된 얼굴의 시각적 요약을 제공하기 위해, `place_detected_faces` 함수는 추출된 얼굴 이미지를 비디오 프레임의 지정된 영역에 오버레이합니다. 이러한 이미지를 적절히 크기 조정하고 배치함으로써 비디오의 주요 콘텐츠가 방해받지 않으면서 각 프레임에서 발견된 모든 얼굴의 관련 정보를 표시합니다.

**오버레이 위치 지정:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (\text{width} - 120, 10 + 110 \times \text{idx})
\]

여기서:
- \( \text{idx} \) = 감지된 얼굴의 인덱스

## 8. **참조 객체 로드 (`load_reference_objects`)**

이 함수는 지정된 디렉토리에서 객체의 참조 이미지를 로드합니다. 이러한 참조 이미지는 객체 인식을 위한 기준으로 작용하여, 애플리케이션이 비디오 프레임에서 감지된 객체를 비교하고 식별할 수 있게 합니다. 이는 객체 추적 및 분류와 같은 작업에 필수적입니다.

**이미지 로드:**

\[
\text{ref\_objects}[ \text{class\_name.lower()} ] = \text{cv2.imread}(\text{file\_path})
\]

## 9. **프레임에서 객체 감지 (`detect_objects_on_frame`)**

YOLOv5 모델을 사용하여, `detect_objects_on_frame` 함수는 각 비디오 프레임 내에서 객체를 감지하고 분류합니다. YOLOv5의 실시간 객체 감지 기능은 여러 객체를 효율적으로 식별하고, 정확한 위치 지정과 분류를 위해 클래스 이름과 경계 상자 좌표를 제공합니다.

**YOLOv5 감지:**

YOLOv5는 이미지를 그리드로 나누고 각 그리드 셀에 대해 경계 상자와 클래스 확률을 예측합니다.

**경계 상자 예측:**

\[
(x, y, w, h, \text{confidence}, \text{class\_scores})
\]

여기서:
- \( x, y \) = 중심 좌표
- \( w, h \) = 너비와 높이
- \( \text{confidence} \) = 객체 신뢰도 점수
- \( \text{class\_scores} \) = 각 클래스에 대한 확률

## 10. **프레임에 참조 객체 배치 (`place_reference_objects`)**

얼굴 배치와 유사하게, 이 함수는 인식된 객체의 이미지를 비디오 프레임에 오버레이합니다. 이러한 참조 객체 이미지를 특정 영역에 배치함으로써, 비디오 전체에서 감지된 객체의 명확한 시각적 표현을 제공하여 처리된 콘텐츠의 정보 가치를 높입니다.

**오버레이 위치 지정:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (20, 10 + 110 \times \text{idx})
\]

여기서:
- \( \text{idx} \) = 감지된 객체의 인덱스

## 11. **비디오 처리 (`process_video`)**

`process_video` 함수는 종합적인 비디오 처리 워크플로를 조직합니다. 입력 비디오의 각 프레임에 대해 얼굴 감지, 포즈 인식, 자막 추가 및 객체 감지를 순차적으로 적용합니다. 처리된 프레임은 모든 주석과 오버레이가 포함된 최종 출력 비디오로 컴파일되어 일관되고 풍부한 시청 경험을 보장합니다.

**워크플로 단계:**
1. **프레임 추출:** `cv2.VideoCapture`를 사용하여 입력 비디오에서 프레임을 읽습니다.
2. **기능 적용:** 각 프레임에 대해 감지 및 주석 기능을 적용합니다.
3. **프레임 컴파일:** `cv2.VideoWriter`를 사용하여 주석이 추가된 프레임을 출력 비디오 파일에 기록합니다.

**프레임 처리 루프:**

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

## 12. **메인 함수 (`main`)**

진입점으로서, `main` 함수는 OptiVerse 애플리케이션의 전체 실행을 관리합니다. 오디오 추출 및 전사를 시작하고, 자막 생성을 처리하며, 비디오 처리 파이프라인을 트리거합니다. 또한, 처리 단계를 추적하고 발생할 수 있는 예외를 처리하기 위한 로깅 메커니즘을 통합하여 원활하고 신뢰할 수 있는 운영을 보장합니다.

**실행 흐름:**
1. **오디오 추출:** `extract_audio` 함수를 호출하여 오디오 트랙을 가져옵니다.
2. **전사:** 추출된 오디오를 `transcribe_audio` 함수에 전달하여 텍스트로 변환합니다.
3. **자막 저장:** 전사된 세그먼트를 `subtitles.txt` 파일에 기록합니다.
4. **포즈 초기화:** MediaPipe의 Pose 솔루션을 초기화합니다.
5. **비디오 처리:** 비디오 경로, 전사 세그먼트 및 포즈 객체를 사용하여 `process_video` 함수를 호출합니다.
6. **로깅:** 처리 중에 발생한 상태 및 오류를 기록합니다.
