![Logo](https://github.com/Solrikk/OptiVerse/blob/main/assets/OpenCV%20-%20result/bee.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/README.md">English</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_RU.md">⭐Russian⭐</a> |
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
    git clone https://github.com/your-username/OptiVerse.git
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

# 🔧 Детальное Описание Функционала

OptiVerse использует набор передовых технологий для обеспечения надежных возможностей обработки видео. Ниже приведён технический обзор каждой функции, выделяющий ключевые компоненты, методологии и используемые формулы.

## 1. **Извлечение Аудио (`extract_audio`)**

Функция `extract_audio` использует библиотеку `ffmpeg` для извлечения аудиопотока из заданного видеофайла. Преобразование аудио в формат WAV обеспечивает совместимость с различными моделями транскрипции. Эта функция обрабатывает создание временных файлов и включает обработку ошибок для эффективного управления любыми проблемами во время процесса извлечения.

## 2. **Транскрипция Аудио (`transcribe_audio`)**

Используя модель Whisper от OpenAI, функция `transcribe_audio` преобразует извлечённое аудио в текст. Надёжные возможности транскрипции Whisper обеспечивают точные и временные сегменты, необходимые для синхронизации субтитров с видео. Процесс транскрипции можно математически описать с использованием **Архитектуры Трансформеров**, где входные аудио признаки преобразуются в последовательность текстовых токенов.

**Формула Трансформера:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Где:
- \( Q \) = Матрица запросов (Query)
- \( K \) = Матрица ключей (Key)
- \( V \) = Матрица значений (Value)
- \( d_k \) = Размерность ключевых векторов

Этот механизм внимания позволяет модели взвешивать релевантность различных частей входного аудио при генерации каждой части транскрибированного текста.

## 3. **Обнаружение и Захват Лиц (`detect_and_capture_faces`)**

Эта функция использует классификатор каскадов Хаара из OpenCV для идентификации и извлечения лиц из каждого кадра видео. Обработка кадров в градациях серого улучшает точность и производительность обнаружения. Обнаруженные лица выделяются рамками и сохраняются как отдельные изображения для потенциального будущего анализа или приложений, таких как распознавание лиц.

**Классификатор Каскадов Хаара:**

Классификатор использует признаки Хаара для обнаружения объектов (лиц) путем сканирования изображения на нескольких масштабах и позициях.

**Расчёт Интегрального Изображения:**

\[
\text{Integral Image}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
\]

Где \( I(i, j) \) — интенсивность пикселя в позиции \( (i, j) \).

Это позволяет быстро вычислять признаки Хаара, обеспечивая эффективное обнаружение лиц.

## 4. **Распознавание Поз (`detect_pose_on_frame`)**

Используя решение Pose от MediaPipe, функция `detect_pose_on_frame` идентифицирует ключевые точки тела в каждом кадре видео. Это позволяет визуализировать позы человека путем наложения скелетных соединений на видео, что особенно полезно для приложений в области анализа движений, спортивного тренинга и анимации.

**Обнаружение Ключевых Точек:**

MediaPipe использует многоступенчатый конвейер, включающий:
1. **Обнаружение Лендмарков:** Идентифицирует ключевые точки тела.
2. **Оценка Позы:** Определяет пространственные отношения между лендмарками.

**Пример Формулы для Расчёта Угла Сустава:**

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)
\]

Где \( \mathbf{u} \) и \( \mathbf{v} \) — векторы, представляющие конечности, а \( \theta \) — угол между ними.

## 5. **Добавление Субтитров (`add_subtitles`)**

Функция `add_subtitles` интегрирует транскрибированный текст в видео путем наложения синхронизированных субтитров. Она сопоставляет текущее время воспроизведения с соответствующими сегментами транскрипции, обеспечивая точное отображение текста, соответствующего произнесённой аудио. Это повышает доступность и улучшает качество просмотра.

**Сопоставление Временных Меток:**

\[
\text{Subtitle Time Window} = [t_{\text{start}}, t_{\text{end}}]
\]

Субтитры отображаются, когда текущее время видео \( t \) удовлетворяет условию:

\[
t_{\text{start}} \leq t \leq t_{\text{end}}
\]

## 6. **Сохранение Извлечённых Лиц (`save_faces`)**

Обнаруженные лица систематически сохраняются в директорию `extracted_faces` с использованием уникальных имен файлов, которые включают номер кадра и индекс лица. Такая организованная система хранения облегчает дальнейшее извлечение и управление изображениями лиц для последующей обработки или анализа.

**Конвенция Именования Файлов:**

\[
\text{filename} = \text{"frame\_"} + \text{frame\_count} + \text{"\_face\_"} + \text{face\_index} + \text{".png"}
\]

## 7. **Размещение Обнаруженных Лиц на Кадре (`place_detected_faces`)**

Для предоставления визуального резюме обнаруженных лиц, функция `place_detected_faces` накладывает извлечённые изображения лиц на назначенную область видеокадра. Путём изменения размера и соответствующего позиционирования этих изображений она гарантирует, что основной контент видео остаётся незатронутым, при этом отображается релевантная информация о лицах.

**Позиционирование Наложения:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (\text{width} - 120, 10 + 110 \times \text{idx})
\]

Где:
- \( \text{idx} \) = Индекс обнаруженного лица

## 8. **Загрузка Справочных Объектов (`load_reference_objects`)**

Эта функция загружает справочные изображения объектов из указанной директории. Эти справочные изображения служат базой для распознавания объектов, позволяя приложению сравнивать и идентифицировать объекты, обнаруженные в кадрах видео. Это необходимо для таких задач, как отслеживание и классификация объектов.

**Загрузка Изображений:**

\[
\text{ref\_objects}[ \text{class\_name.lower()} ] = \text{cv2.imread}(\text{file\_path})
\]

## 9. **Обнаружение Объектов на Кадре (`detect_objects_on_frame`)**

Используя модель YOLOv5, функция `detect_objects_on_frame` обнаруживает и классифицирует объекты в каждом кадре видео. Возможности YOLOv5 по обнаружению объектов в реальном времени обеспечивают эффективную идентификацию нескольких объектов, предоставляя их названия классов и координаты ограничивающих рамок для точной локализации и категоризации.

**Обнаружение YOLOv5:**

YOLOv5 делит изображение на сетку и предсказывает ограничивающие рамки и вероятности классов для каждой ячейки сетки.

**Предсказание Ограничивающих Рамок:**

\[
(x, y, w, h, \text{confidence}, \text{class\_scores})
\]

Где:
- \( x, y \) = Центровые координаты
- \( w, h \) = Ширина и высота
- \( \text{confidence} \) = Оценка уверенности в обнаружении объекта
- \( \text{class\_scores} \) = Вероятности для каждого класса

## 10. **Размещение Справочных Объектов на Кадре (`place_reference_objects`)**

Похожим образом на размещение лиц, эта функция накладывает изображения распознанных объектов на видеокадр. Путём позиционирования этих справочных изображений объектов в определённой области она предоставляет чёткое визуальное представление обнаруженных объектов на протяжении всего видео, повышая информационную ценность обработанного контента.

**Позиционирование Наложения:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (20, 10 + 110 \times \text{idx})
\]

Где:
- \( \text{idx} \) = Индекс обнаруженного объекта

## 11. **Обработка Видео (`process_video`)**

Функция `process_video` организует комплексный рабочий процесс обработки видео. Она последовательно применяет обнаружение лиц, распознавание поз, добавление субтитров и обнаружение объектов к каждому кадру входного видео. Обработанные кадры затем компилируются в итоговое выходное видео, которое включает все аннотации и наложения, обеспечивая целостный и обогащённый опыт просмотра.

**Этапы Рабочего Процесса:**
1. **Извлечение Кадров:** Чтение кадров из входного видео с помощью `cv2.VideoCapture`.
2. **Применение Функций:** Применение функций обнаружения и аннотации к каждому кадру.
3. **Компиляция Кадров:** Запись аннотированных кадров в выходной видеофайл с помощью `cv2.VideoWriter`.

**Цикл Обработки Кадров:**

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

## 12. **Главная Функция (`main`)**

Служащая точкой входа, функция `main` управляет общей работой приложения OptiVerse. Она инициирует извлечение и транскрипцию аудио, обрабатывает генерацию субтитров и запускает конвейер обработки видео. Дополнительно, она включает механизмы логирования для отслеживания этапов обработки и обработки любых возникших исключений, обеспечивая плавную и надёжную работу.

**Поток Выполнения:**
1. **Извлечение Аудио:** Вызов функции `extract_audio` для получения аудиодорожки.
2. **Транскрипция:** Передача извлечённого аудио в функцию `transcribe_audio` для преобразования в текст.
3. **Сохранение Субтитров:** Запись сегментов транскрипции в файл `subtitles.txt`.
4. **Инициализация Позы:** Инициализация решения Pose от MediaPipe.
5. **Обработка Видео:** Вызов функции `process_video` с путем к видео, сегментами транскрипции и объектом позы.
6. **Логирование:** Запись статуса и любых ошибок, возникших во время обработки.

**Обработка Ошибок:**

```python
try:
    perform processing steps
except Exception as e:
    log the error message
```
