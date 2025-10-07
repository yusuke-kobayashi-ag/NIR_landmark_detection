## プロジェクト構造

```
landmark_NIR_ver4/
├── main.py                      # メイン実行スクリプト
├── config.py                    # 設定クラス
├── constants.py                 # 定数定義
├── data_models.py              # データモデル（dataclass）
├── logger.py                    # ログ管理
├── image_processor.py          # 画像前処理
├── landmark_detector.py        # ランドマーク検出
├── file_handler.py             # ファイル操作
├── processor.py                # 画像処理実行
├── directory_processor.py      # ディレクトリ処理
├── create_folder_list.py       # フォルダリスト作成ツール
├── requirements.txt            # 依存パッケージ
└── shape_predictor_68_face_landmarks.dat  # dlibモデル
```

## 使用方法

## 仮想環境の作成と有効化

```bash
# 仮想環境の作成
conda create -n nir_landmark

# 仮想環境の有効化
conda activate nir_landmark

# 必要なパッケージのインストール
conda install -c conda-forge dlib
pip install -r requirements.txt
```

## 実行方法

### GUIツールを使用する場合（推奨）

```bash
python create_folder_list.py
```

1. GUIが起動したら、以下の操作が可能です：
   - 「フォルダを追加」: 個別にフォルダを選択して追加
   - 「フォルダを一括追加」: 親フォルダを選択し、その中の.npyファイルを含むフォルダを一括追加
   - 「選択したフォルダを削除」: リストから不要なフォルダを削除

2. フォルダ名でフィルタリングする場合：
   - 上部の「フォルダ名フィルター」に検索文字列を入力（カンマ区切りで複数指定可能）
   - 「フォルダを一括追加」を押すと、フィルター条件に一致するフォルダのみが追加されます

3. 「リストを保存して処理開始」ボタンを押すと：
   - フォルダリストが保存され
   - 自動的に処理が開始されます

### コマンドラインから直接実行する場合

```bash
# 複数のフォルダを直接指定
python main.py --dirs folder1 folder2 folder3 --mode [normal|high]

# フォルダリストを使用
python main.py --list folder_list.txt --mode [normal|high]
```
*normalの場合はかかなくてもだいじょうぶ

### 引数

- `--dirs`: 処理するNIR画像（.npy形式）が含まれるディレクトリのパス（複数指定可能）
- `--list`: 処理するディレクトリのパスが記載されたテキストファイル
- `--mode`: 検出モード
  - `normal`: 通常モード（アップサンプリング3回まで）
  - `high`: 高精度モード（アップサンプリング4回まで）

### 出力

処理結果は以下のディレクトリに保存されます：

- `processed_data/[入力ディレクトリ名]/orignorm/`: 正規化された元画像
- `processed_data/[入力ディレクトリ名]/processed/`: 前処理済み画像
- `processed_data/[入力ディレクトリ名]/landmarks/`: 検出されたランドマーク
- `processed_data/[入力ディレクトリ名]/comparisons/`: 比較画像（元画像、処理後画像、ランドマーク付き画像）
- `processed_data/detection_results.txt`: 検出結果のサマリー
- `processed_data/[入力ディレクトリ名]/not_detected.txt`: 検出失敗した画像のリスト

## モジュール説明

### コアモジュール

- **`constants.py`**: 定数定義（ランドマークインデックスなど）
- **`config.py`**: 設定クラス（モデルパス、画像処理パラメータ、テンプレートランドマーク）
- **`data_models.py`**: データクラス定義（処理結果、検出情報など）
- **`logger.py`**: ログ管理クラス（エラーログ）

### 処理モジュール

- **`image_processor.py`**: 画像の前処理（正規化、ガンマ補正、CLAHE、フィルタリング）
- **`landmark_detector.py`**: 顔ランドマーク検出（dlibを使用）
- **`file_handler.py`**: ファイル操作（ディレクトリ設定、ファイル保存、可視化）
- **`processor.py`**: 個別画像処理（画像読み込み→前処理→検出→保存）
- **`directory_processor.py`**: ディレクトリ処理（複数画像の並列処理）

### 実行モジュール

- **`main.py`**: メインスクリプト（コマンドライン引数解析、処理実行）
- **`create_folder_list.py`**: GUIツール（フォルダ選択、フィルタリング）

## 注意事項

- 入力画像は.npy形式である必要があります
- 高精度モード（`--mode high`）は処理時間が長くなりますが、検出精度が向上する可能性があります
- メモリ使用量は入力画像のサイズと数に依存します
- 各モジュールは独立しており、保守・拡張が容易です
