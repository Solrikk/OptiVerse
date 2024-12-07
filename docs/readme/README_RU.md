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

OptiVerse — это мощное приложение для обработки видео, объединяющее передовые технологии компьютерного зрения и машинного обучения. С помощью OptiVerse вы можете автоматически извлекать и транскрибировать аудио, обнаруживать лица и позы, распознавать объекты, а также добавлять субтитры и аннотации к видеофайлам.

## Основные возможности

- **Извлечение аудио**: Автоматически извлекает аудиодорожку из видеофайла.
- **Транскрипция аудио**: Преобразует речь из аудио в текст с использованием модели Whisper от OpenAI.
- **Обнаружение лиц**: Идентифицирует и извлекает лица из каждого кадра видео.
- **Распознавание поз**: Определяет позы и ключевые точки тела с помощью MediaPipe.
- **Распознавание объектов**: Использует модель YOLOv5 для обнаружения и классификации объектов в кадре.
- **Добавление субтитров**: Автоматически генерирует и накладывает субтитры на видео на основе транскрибированного текста.
- **Аннотации и оверлеи**: Добавляет рамки вокруг обнаруженных лиц и объектов, а также отображает позы и дополнительные визуальные элементы.
- **Сохранение результатов**: Экспортирует обработанное видео с аннотациями и сохраняет извлечённые лица и объекты в отдельные директории.

## Установка

1. **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/ваш-логин/OptiVerse.git
    cd OptiVerse
    ```

2. **Создайте и активируйте виртуальное окружение (опционально):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Для Linux/Mac
    venv\Scripts\activate     # Для Windows
    ```

3. **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

    **Примечание:** Убедитесь, что у вас установлены все необходимые библиотеки, включая `torch`, `whisper`, `ffmpeg`, `opencv-python`, `mediapipe`, и `yolov5`.

4. **Скачайте модель YOLOv5:**
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```

    Поместите файл модели `yolov5l.pt` в корневую директорию проекта или укажите путь к модели в коде.

## Использование

1. **Подготовьте видеофайл:**
    Поместите видео, которое вы хотите обработать, в корневую директорию проекта и обновите путь к файлу в функции `main()`:
    ```python
    video_path = 'ваше_видео.mp4'
    ```

2. **Запустите скрипт:**
    ```bash
    python optiverse.py
    ```

3. **Результаты:**
    - Обработанное видео будет сохранено под именем `output_with_faces_and_pose_and_subtitles.mp4`.
    - Извлечённые лица сохранятся в директории `extracted_faces`.
    - Извлечённые объекты сохранятся в директории `extracted_object`.
    - Субтитры будут сохранены в файле `subtitles.txt`.

## Структура проекта

- `optiverse.py` — основной скрипт обработки видео.
- `yolov5/` — директория с моделью YOLOv5.
- `extracted_faces/` — директория для сохранения извлечённых лиц.
- `extracted_object/` — директория для сохранения извлечённых объектов.
- `requirements.txt` — список зависимостей проекта.

## Зависимости

- Python 3.7+
- PyTorch
- Whisper
- FFmpeg
- OpenCV
- MediaPipe
- YOLOv5



