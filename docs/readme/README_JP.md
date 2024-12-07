![Logo](https://github.com/Solrikk/OptiVerse/blob/main/assets/OpenCV%20-%20result/bee.jpg)

<div align="center">
  <h3>
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/README.md">English</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_RU.md">Russian</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_GE.md">German</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme//README_JP.md">⭐Japanese⭐</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_KR.md">Korean</a> |
    <a href="https://github.com/Solrikk/OptiVerse/blob/main/docs/readme/README_CN.md">Chinese</a>
  </h3>
</div>

-----------------

# OptiVerse

OptiVerseは、先進的なコンピュータビジョンおよび機械学習技術を組み合わせた強力なビデオ処理アプリケーションです。OptiVerseを使用すると、ビデオファイルから音声を自動的に抽出および転写し、顔やポーズを検出し、オブジェクトを認識するとともに、字幕や注釈をビデオに追加することができます。

## 主な機能

- **音声抽出**: ビデオファイルから音声トラックを自動的に抽出します。
- **音声転写**: OpenAIのWhisperモデルを使用して、音声からテキストに変換します。
- **顔検出**: ビデオの各フレームから顔を識別および抽出します。
- **ポーズ認識**: MediaPipeを使用して、ポーズや主要な身体のポイントを判定します。
- **オブジェクト認識**: YOLOv5モデルを活用して、フレーム内のオブジェクトを検出および分類します。
- **字幕追加**: 転写されたテキストに基づいて、ビデオに自動的に字幕を生成および重ねます。
- **注釈とオーバーレイ**: 検出された顔やオブジェクトに境界ボックスを追加し、ポーズや追加の視覚要素を表示します。
- **結果の保存**: 注釈付きの処理済みビデオをエクスポートし、抽出された顔やオブジェクトを別々のディレクトリに保存します。

## インストール

1. **リポジトリをクローンする:**
    ```bash
    git clone https://github.com/your-username/OptiVerse.git
    cd OptiVerse
    ```

2. **仮想環境を作成およびアクティブ化する（オプション）:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Macの場合
    venv\Scripts\activate     # Windowsの場合
    ```

3. **依存関係をインストールする:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **注意:** `torch`、`whisper`、`ffmpeg`、`opencv-python`、`mediapipe`、および`yolov5`を含むすべての必要なライブラリがインストールされていることを確認してください。

4. **YOLOv5モデルをダウンロードする:**
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```
    
    モデルファイル`yolov5l.pt`をプロジェクトのルートディレクトリに配置するか、コード内でモデルへのパスを指定してください。

## 使用方法

1. **ビデオファイルを準備する:**
    処理したいビデオをプロジェクトのルートディレクトリに配置し、`main()`関数内のファイルパスを更新します。
    ```python
    video_path = 'your_video.mp4'
    ```

2. **スクリプトを実行する:**
    ```bash
    python optiverse.py
    ```

3. **結果:**
    - 処理済みビデオは`output_with_faces_and_pose_and_subtitles.mp4`として保存されます。
    - 抽出された顔は`extracted_faces`ディレクトリに保存されます。
    - 抽出されたオブジェクトは`extracted_object`ディレクトリに保存されます。
    - 字幕は`subtitles.txt`ファイルに保存されます。

# 🔧 詳細な機能説明

OptiVerseは、堅牢なビデオ処理機能を提供するために、複数の先進技術を統合しています。以下は、各機能の技術的概要であり、主要なコンポーネント、手法、および関連する数式を強調しています。

## 1. **音声抽出 (`extract_audio`)**

`extract_audio`関数は、`ffmpeg`ライブラリを使用して指定されたビデオファイルからオーディオストリームを抽出します。オーディオをWAV形式に変換することで、さまざまな転写モデルとの互換性を確保します。この関数は、一時ファイルの作成を処理し、抽出プロセス中に発生する問題を効率的に管理するためのエラーハンドリングを含みます。

## 2. **音声転写 (`transcribe_audio`)**

OpenAIのWhisperモデルを活用し、`transcribe_audio`関数は抽出されたオーディオをテキストに変換します。Whisperの堅牢な転写能力は、正確でタイムスタンプ付きのセグメントを提供し、ビデオとの字幕の同期に不可欠です。転写プロセスは、入力オーディオ特徴量をテキストトークンのシーケンスに変換する**トランスフォーマーアーキテクチャ**を使用して数学的に説明できます。

**トランスフォーマーの式:**

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

ここで:
- \( Q \) = クエリ行列 (Query matrix)
- \( K \) = キー行列 (Key matrix)
- \( V \) = バリュー行列 (Value matrix)
- \( d_k \) = キーベクトルの次元

このアテンションメカニズムにより、モデルは転写テキストの各部分を生成する際に、入力オーディオの異なる部分の関連性を重み付けすることができます。

## 3. **顔検出とキャプチャ (`detect_and_capture_faces`)**

この機能は、OpenCVのHaarカスケード分類器を使用してビデオの各フレームから顔を識別および抽出します。フレームをグレースケールで処理することで、検出の精度とパフォーマンスを向上させます。検出された顔にはバウンディングボックスが描画され、顔画像が個別に保存され、将来の分析や顔認識などのアプリケーションに利用できます。

**Haarカスケード分類器:**

分類器は、複数のスケールと位置で画像をスキャンすることで、Haar様の特徴を使用してオブジェクト（顔）を検出します。

**積分画像の計算:**

\[
\text{Integral Image}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
\]

ここで \( I(i, j) \) は位置 \( (i, j) \) におけるピクセルの強度です。

これにより、Haar特徴の迅速な計算が可能となり、効率的な顔検出が実現します。

## 4. **ポーズ認識 (`detect_pose_on_frame`)**

MediaPipeのPoseソリューションを活用し、`detect_pose_on_frame`関数はビデオの各フレーム内で主要な身体ランドマークを識別します。これにより、ビデオに骨格的な接続をオーバーレイすることで人間のポーズを視覚化することができ、動作分析、スポーツトレーニング、およびアニメーションなどのアプリケーションに特に有用です。

**キーポイント検出:**

MediaPipeは、以下を含むマルチステージパイプラインを使用します:
1. **ランドマーク検出:** 主要な身体ポイントを識別します。
2. **ポーズ推定:** ランドマーク間の空間的関係を決定します。

**関節角度計算の例式:**

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)
\]

ここで \( \mathbf{u} \) と \( \mathbf{v} \) は手足を表すベクトルであり、\( \theta \) はその間の角度です。

## 5. **字幕追加 (`add_subtitles`)**

`add_subtitles`関数は、転写されたテキストをビデオに統合し、同期された字幕をオーバーレイします。現在の再生時間を対応する転写セグメントと照合し、字幕が発話された音声に正確に対応するようにします。これにより、アクセシビリティが向上し、視聴体験が向上します。

**タイムスタンプの照合:**

\[
\text{Subtitle Time Window} = [t_{\text{start}}, t_{\text{end}}]
\]

字幕は、現在のビデオ時間 \( t \) が以下の条件を満たす場合に表示されます:

\[
t_{\text{start}} \leq t \leq t_{\text{end}}
\]

## 6. **抽出された顔の保存 (`save_faces`)**

検出された顔は、フレーム番号と顔のインデックスを含む一意のファイル名を使用して`extracted_faces`ディレクトリに体系的に保存されます。この整理された保存により、さらなる処理や分析のために顔画像を容易に取得および管理できます。

**ファイル名の規約:**

\[
\text{filename} = \text{"frame\_"} + \text{frame\_count} + \text{"\_face\_"} + \text{face\_index} + \text{".png"}
\]

## 7. **フレーム上に検出された顔を配置する (`place_detected_faces`)**

検出された顔の視覚的な概要を提供するために、`place_detected_faces`関数は抽出された顔画像をビデオフレームの指定された領域にオーバーレイします。これらの画像を適切にリサイズおよび配置することで、ビデオの主要なコンテンツが遮られることなく、各フレームで見つかったすべての顔の関連情報が表示されます。

**オーバーレイの位置決め:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (\text{width} - 120, 10 + 110 \times \text{idx})
\]

ここで:
- \( \text{idx} \) = 検出された顔のインデックス

## 8. **参照オブジェクトの読み込み (`load_reference_objects`)**

この関数は、指定されたディレクトリからオブジェクトの参照画像を読み込みます。これらの参照画像は、ビデオフレームで検出されたオブジェクトを比較および識別するためのベースラインとして機能します。これにより、オブジェクトの追跡や分類などのタスクが可能になります。

**画像の読み込み:**

\[
\text{ref\_objects}[ \text{class\_name.lower()} ] = \text{cv2.imread}(\text{file\_path})
\]

## 9. **フレーム上のオブジェクト検出 (`detect_objects_on_frame`)**

YOLOv5モデルを使用して、`detect_objects_on_frame`関数は各ビデオフレーム内のオブジェクトを検出および分類します。YOLOv5のリアルタイムオブジェクト検出機能により、複数のオブジェクトを効率的に識別し、そのクラス名および境界ボックスの座標を提供して正確な位置特定と分類を可能にします。

**YOLOv5検出:**

YOLOv5は画像をグリッドに分割し、各グリッドセルごとに境界ボックスとクラス確率を予測します。

**境界ボックスの予測:**

\[
(x, y, w, h, \text{confidence}, \text{class\_scores})
\]

ここで:
- \( x, y \) = センター座標
- \( w, h \) = 幅と高さ
- \( \text{confidence} \) = オブジェクトの信頼度スコア
- \( \text{class\_scores} \) = 各クラスの確率

## 10. **フレーム上に参照オブジェクトを配置する (`place_reference_objects`)**

顔の配置と同様に、この関数は認識されたオブジェクトの画像をビデオフレームにオーバーレイします。これらの参照オブジェクト画像を特定の領域に配置することで、ビデオ全体で検出されたオブジェクトの明確な視覚的表現を提供し、処理されたコンテンツの情報価値を高めます。

**オーバーレイの位置決め:**

\[
(x_{\text{offset}}, y_{\text{offset}}) = (20, 10 + 110 \times \text{idx})
\]

ここで:
- \( \text{idx} \) = 検出されたオブジェクトのインデックス

## 11. **ビデオ処理 (`process_video`)**

`process_video`関数は、包括的なビデオ処理ワークフローを統括します。入力ビデオの各フレームに対して、顔検出、ポーズ認識、字幕追加、およびオブジェクト検出を順次適用します。処理されたフレームは、すべての注釈とオーバーレイを含む最終的な出力ビデオにコンパイルされ、統一感のある豊かな視聴体験を提供します。

**ワークフローステップ:**
1. **フレームの抽出:** `cv2.VideoCapture`を使用して入力ビデオからフレームを読み取ります。
2. **機能の適用:** 各フレームに対して検出および注釈機能を適用します。
3. **フレームのコンパイル:** `cv2.VideoWriter`を使用して注釈付きフレームを出力ビデオファイルに書き込みます。

**フレーム処理ループ:**

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

## 12. **メイン関数 (`main`)**

エントリーポイントとして機能する`main`関数は、OptiVerseアプリケーション全体の実行を管理します。音声の抽出と転写を開始し、字幕生成を処理し、ビデオ処理パイプラインをトリガーします。さらに、処理ステージを追跡し、発生する可能性のある例外を処理するためのログ記録メカニズムを統合し、スムーズで信頼性の高い運用を確保します。

**実行フロー:**
1. **音声抽出:** `extract_audio`を呼び出してオーディオトラックを取得します。
2. **転写:** 抽出されたオーディオを`transcribe_audio`に渡してテキスト変換を行います。
3. **字幕の保存:** 転写セグメントを`subtitles.txt`ファイルに書き込みます。
4. **ポーズの初期化:** MediaPipeのPoseソリューションを初期化します。
5. **ビデオ処理:** ビデオパス、転写セグメント、およびポーズオブジェクトを使用して`process_video`を呼び出します。
6. **ログ記録:** 処理中に発生したステータスおよびエラーを記録します。
