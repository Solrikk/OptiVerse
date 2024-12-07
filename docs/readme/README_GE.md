![Logo](https://github.com/Solrikk/OptiVerse/blob/main/assets/OpenCV%20-%20result/bee.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/README.md">English</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_GE.md">⭐German⭐</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme//README_JP.md">Japanese</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

-----------------

# OptiVerse

OptiVerse ist eine leistungsstarke Videoverarbeitungsanwendung, die fortschrittliche Technologien der Computer Vision und des maschinellen Lernens kombiniert. Mit OptiVerse können Sie automatisch Audio extrahieren und transkribieren, Gesichter und Posen erkennen, Objekte identifizieren sowie Untertitel und Anmerkungen zu Videodateien hinzufügen.

## Hauptfunktionen

- **Audioextraktion**: Extrahiert automatisch die Audiospur aus einer Videodatei.
- **Audiotranskription**: Konvertiert Sprache aus Audio in Text mithilfe des Whisper-Modells von OpenAI.
- **Gesichtserkennung**: Identifiziert und extrahiert Gesichter aus jedem Frame des Videos.
- **Posenerkennung**: Bestimmt Posen und wichtige Körperpunkte mithilfe von MediaPipe.
- **Objekterkennung**: Nutzt das YOLOv5-Modell zur Erkennung und Klassifizierung von Objekten im Frame.
- **Hinzufügen von Untertiteln**: Generiert automatisch Untertitel und überlagert sie basierend auf dem transkribierten Text auf das Video.
- **Anmerkungen und Overlays**: Fügt Begrenzungsrahmen um erkannte Gesichter und Objekte hinzu und zeigt Posen sowie zusätzliche visuelle Elemente an.
- **Speichern von Ergebnissen**: Exportiert das verarbeitete Video mit Anmerkungen und speichert extrahierte Gesichter und Objekte in separaten Verzeichnissen.

## Installation

1. **Repository klonen:**
    ```bash
    git clone https://github.com/your-username/OptiVerse.git
    cd OptiVerse
    ```

2. **Virtuelle Umgebung erstellen und aktivieren (optional):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Für Linux/Mac
    venv\Scripts\activate     # Für Windows
    ```

3. **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **Hinweis:** Stellen Sie sicher, dass alle erforderlichen Bibliotheken installiert sind, einschließlich `torch`, `whisper`, `ffmpeg`, `opencv-python`, `mediapipe` und `yolov5`.

4. **YOLOv5-Modell herunterladen:**
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```
    
    Platzieren Sie die Modell-Datei `yolov5l.pt` im Stammverzeichnis des Projekts oder geben Sie den Pfad zum Modell im Code an.

## Verwendung

1. **Videodatei vorbereiten:**
    Platzieren Sie das Video, das Sie verarbeiten möchten, im Stammverzeichnis des Projekts und aktualisieren Sie den Dateipfad in der Funktion `main()`:
    ```python
    video_path = 'your_video.mp4'
    ```

2. **Skript ausführen:**
    ```bash
    python optiverse.py
    ```

3. **Ergebnisse:**
    - Das verarbeitete Video wird als `output_with_faces_and_pose_and_subtitles.mp4` gespeichert.
    - Extrahierte Gesichter werden im Verzeichnis `extracted_faces` gespeichert.
    - Extrahierte Objekte werden im Verzeichnis `extracted_object` gespeichert.
    - Untertitel werden in der Datei `subtitles.txt` gespeichert.

# 🔧 Detaillierte Funktionsbeschreibung

OptiVerse nutzt eine Vielzahl fortschrittlicher Technologien, um robuste Videoverarbeitungsfähigkeiten bereitzustellen. Nachfolgend finden Sie einen technischen Überblick über jede Funktion, der die wichtigsten Komponenten, Methodologien und relevanten Formeln hervorhebt.

## 1. **Audioextraktion (`extract_audio`)**

Die Funktion `extract_audio` verwendet die Bibliothek `ffmpeg`, um den Audiostream aus einer gegebenen Videodatei zu extrahieren. Durch die Konvertierung des Audios in das WAV-Format wird die Kompatibilität mit verschiedenen Transkriptionsmodellen sichergestellt. Diese Funktion kümmert sich um die Erstellung temporärer Dateien und beinhaltet eine Fehlerbehandlung, um Probleme während des Extraktionsprozesses effizient zu verwalten.

## 2. **Audiotranskription (`transcribe_audio`)**

Unter Verwendung des Whisper-Modells von OpenAI konvertiert die Funktion `transcribe_audio` das extrahierte Audio in Text. Die robusten Transkriptionsfähigkeiten von Whisper liefern genaue und zeitlich abgestimmte Segmente, die für die Synchronisierung von Untertiteln mit dem Video unerlässlich sind. Der Transkriptionsprozess kann mathematisch mit der **Transformer-Architektur** beschrieben werden, bei der die Eingangs-Audiofeatures in eine Sequenz von Texttokens transformiert werden.

**Transformator-Gleichung:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Dabei gilt:
- \( Q \) = Abfrage-Matrix (Query)
- \( K \) = Schlüssel-Matrix (Key)
- \( V \) = Wert-Matrix (Value)
- \( d_k \) = Dimension der Schlüssel-Vektoren

Dieser Aufmerksamkeitsmechanismus ermöglicht es dem Modell, die Relevanz verschiedener Teile des Eingangs-Audios zu gewichten, wenn jeder Teil des transkribierten Textes generiert wird.

## 3. **Gesichtserkennung und -erfassung (`detect_and_capture_faces`)**

Diese Funktion verwendet den Haar-Kaskaden-Klassifikator von OpenCV, um Gesichter in jedem Videoframe zu identifizieren und zu extrahieren. Durch die Verarbeitung der Frames in Graustufen verbessert die Funktion die Genauigkeit und Leistung der Erkennung. Erkannte Gesichter werden mit Begrenzungsrahmen hervorgehoben und als separate Bilddateien gespeichert, um eine zukünftige Analyse oder Anwendungen wie die Gesichtserkennung zu ermöglichen.

**Haar-Kaskaden-Klassifikator:**

Der Klassifikator verwendet Haar-ähnliche Merkmale, um Objekte (Gesichter) zu erkennen, indem er das Bild in mehreren Maßstäben und Positionen scannt.

**Berechnung des Integralbildes:**

\[
\text{Integral Image}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
\]

Dabei ist \( I(i, j) \) die Pixelintensität an der Position \( (i, j) \).

Dies ermöglicht eine schnelle Berechnung der Haar-Merkmale und somit eine effiziente Gesichtserkennung.

## 4. **Posenerkennung (`detect_pose_on_frame`)**

Unter Nutzung der Pose-Lösung von MediaPipe identifiziert die Funktion `detect_pose_on_frame` Schlüssel-Körper-Landmarks in jedem Videoframe. Dies ermöglicht die Visualisierung menschlicher Posen durch Überlagerung skelettaler Verbindungen auf dem Video, was besonders nützlich für Anwendungen in der Bewegungsanalyse, im Sporttraining und in der Animation ist.

**Erkennung von Schlüsselpunktlandmarks:**

MediaPipe verwendet eine mehrstufige Pipeline, die Folgendes umfasst:
1. **Landmark-Erkennung:** Identifiziert Schlüssel-Körperpunkte.
2. **Pose-Schätzung:** Bestimmt die räumlichen Beziehungen zwischen den Landmarks.

**Beispielgleichung zur Berechnung des Gelenkwinkels:**

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)
\]

Dabei sind \( \mathbf{u} \) und \( \mathbf{v} \) Vektoren, die Gliedmaßen darstellen, und \( \theta \) ist der Winkel zwischen ihnen.

## 5. **Hinzufügen von Untertiteln (`add_subtitles`)**

Die Funktion `add_subtitles` integriert den transkribierten Text in das Video, indem sie synchronisierte Untertitel überlagert. Sie stimmt die aktuelle Wiedergabezeit mit den entsprechenden Transkriptionssegmenten ab und stellt sicher, dass die Untertitel den gesprochenen Audioinhalten genau entsprechen. Dies verbessert die Zugänglichkeit und bietet ein besseres Seherlebnis.

**Zeitstempel-Abgleich:**

\[
\text{Subtitle Time Window} = [t_{\text{start}}, t_{\text{end}}]
\]

Untertitel werden angezeigt, wenn die aktuelle Videowiedergabezeit \( t \) die Bedingung erfüllt:

\[
t_{\text{start}} \leq t \leq t_{\text{end}}
\]

## 6. **Speichern von extrahierten Gesichtern (`save_faces`)**

Erkannte Gesichter werden systematisch im Verzeichnis `extracted_faces` mit eindeutigen Dateinamen gespeichert, die die Frame-Nummer und den Gesichtsindex enthalten. Diese organisierte Speicherung erleichtert das spätere Abrufen und Verwalten von Gesichtsabbildungen für weitere Verarbeitungen oder Analysen.

**Dateinamenskonvention:**

\[
\text{filename} = \text{"frame\_"} + \text{frame\_count} + \text{"\_face\_"} + \text{face\_index} + \text{".png"}
\]

## 7. **Platzierung erkannter Gesichter im Frame (`place_detected_faces`)**

Um eine visuelle Zusammenfassung der erkannten Gesichter bereitzustellen, überlagert die Funktion `place_detected_faces` die extrahierten Gesichtsabbildungen in einem festgelegten Bereich des Videoframes. Durch die entsprechende Größenanpassung und Positionierung dieser Abbildungen wird sichergestellt, dass der Hauptinhalt des Videos nicht verdeckt wird, während dennoch relevante Gesichtsinformationen angezeigt werden.

**Positionierung der Überlagerung:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (\text{width} - 120, 10 + 110 \times \text{idx})
\]

Dabei gilt:
- \( \text{idx} \) = Index des erkannten Gesichts

## 8. **Laden von Referenzobjekten (`load_reference_objects`)**

Diese Funktion lädt Referenzbilder von Objekten aus einem angegebenen Verzeichnis. Diese Referenzbilder dienen als Grundlage für die Objekterkennung, sodass die Anwendung Objekte, die in den Videoframes erkannt werden, vergleichen und identifizieren kann. Dies ist für Aufgaben wie Objektverfolgung und -klassifizierung unerlässlich.

**Bildladen:**

\[
\text{ref\_objects}[ \text{class\_name.lower()} ] = \text{cv2.imread}(\text{file\_path})
\]

## 9. **Objekterkennung im Frame (`detect_objects_on_frame`)**

Unter Verwendung des YOLOv5-Modells erkennt und klassifiziert die Funktion `detect_objects_on_frame` Objekte in jedem Videoframe. Die Echtzeit-Objekterkennung von YOLOv5 ermöglicht eine effiziente Identifizierung mehrerer Objekte und liefert deren Klassennamen sowie die Koordinaten der Begrenzungsrahmen für eine präzise Lokalisierung und Kategorisierung.

**YOLOv5-Erkennung:**

YOLOv5 teilt das Bild in ein Gitter auf und sagt Begrenzungsrahmen sowie Klassenwahrscheinlichkeiten für jede Gitterzelle voraus.

**Vorhersage von Begrenzungsrahmen:**

\[
(x, y, w, h, \text{confidence}, \text{class\_scores})
\]

Dabei gilt:
- \( x, y \) = Zentrumskoordinaten
- \( w, h \) = Breite und Höhe
- \( \text{confidence} \) = Objektwahrscheinlichkeitswert
- \( \text{class\_scores} \) = Wahrscheinlichkeiten für jede Klasse

## 10. **Platzierung von Referenzobjekten im Frame (`place_reference_objects`)**

Ähnlich wie bei der Platzierung von Gesichtern überlagert diese Funktion Bilder erkannter Objekte auf das Videoframe. Durch die Positionierung dieser Referenzobjektbilder in einem bestimmten Bereich bietet sie eine klare visuelle Darstellung der im gesamten Video erkannten Objekte und erhöht so den Informationswert des verarbeiteten Inhalts.

**Positionierung der Überlagerung:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (20, 10 + 110 \times \text{idx})
\]

Dabei gilt:
- \( \text{idx} \) = Index des erkannten Objekts

## 11. **Videoverarbeitung (`process_video`)**

Die Funktion `process_video` organisiert den umfassenden Arbeitsablauf der Videoverarbeitung. Sie wendet nacheinander Gesichtserkennung, Posenerkennung, Hinzufügen von Untertiteln und Objekterkennung auf jeden Frame des Eingabevideos an. Die verarbeiteten Frames werden dann zu einem endgültigen Ausgabevideo kompiliert, das alle Anmerkungen und Überlagerungen enthält und so ein zusammenhängendes und bereichertes Seherlebnis bietet.

**Arbeitsablaufsschritte:**
1. **Frame-Extraktion:** Lesen von Frames aus dem Eingabevideo mit `cv2.VideoCapture`.
2. **Anwendung von Funktionen:** Anwenden von Erkennungs- und Annotationsfunktionen auf jeden Frame.
3. **Frame-Kompilation:** Schreiben der annotierten Frames in eine Ausgabedatei mit `cv2.VideoWriter`.

**Frame-Verarbeitungsschleife:**

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

## 12. **Hauptfunktion (`main`)**

Als Einstiegspunkt dient die Funktion `main`, die den gesamten Ablauf der Anwendung OptiVerse verwaltet. Sie initiiert die Audioextraktion und -transkription, bearbeitet die Generierung von Untertiteln und startet die Videoverarbeitungspipeline. Zusätzlich integriert sie Protokollierungsmechanismen, um die Verarbeitungsschritte zu verfolgen und auftretende Ausnahmen zu handhaben, wodurch ein reibungsloser und zuverlässiger Betrieb gewährleistet wird.

**Ausführungsablauf:**
1. **Audioextraktion:** Aufruf von `extract_audio`, um die Audiospur abzurufen.
2. **Transkription:** Übergabe des extrahierten Audios an `transcribe_audio` zur Textkonvertierung.
3. **Speichern der Untertitel:** Schreiben der Transkriptionssegmente in die Datei `subtitles.txt`.
4. **Poseninitialisierung:** Initialisierung der Pose-Lösung von MediaPipe.
5. **Videoverarbeitung:** Aufruf von `process_video` mit dem Videopfad, den Transkriptionssegmenten und dem Pose-Objekt.
6. **Protokollierung:** Aufzeichnung des Status und eventueller Fehler während der Verarbeitung.
