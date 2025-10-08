"""単一画像からランドマーク座標を返す推論ユーティリティ"""

import os
from typing import Optional, Tuple, Union

import dlib
import numpy as np

from config import Config
from image_processor import preprocess_image
from landmark_detector import detect_landmarks


def _ensure_predictor(model_path: str) -> dlib.shape_predictor:
    """shape_predictor のロードを行う。
    
    Raises:
        FileNotFoundError: 学習済みモデルが存在しない場合
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")
    return dlib.shape_predictor(model_path)


def landmarks_from_array(
    img: np.ndarray,
    config: Optional[Config] = None,
    predictor: Optional[dlib.shape_predictor] = None,
) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
    """入力画像(np.ndarray)から68点ランドマークを返す。

    Args:
        img: 入力画像 (2D or 3D)。3D の場合はグレースケール変換される。
        config: 設定。未指定時は `Config()` を使用。
        predictor: 既にロード済みの dlib.shape_predictor。未指定時は自動ロード。

    Returns:
        (landmarks, is_detected, bounding_box)
        - landmarks: np.ndarray[int32] 形状 (68, 2)。検出失敗時はテンプレートを返す。
        - is_detected: 検出に成功したかどうか
        - bounding_box: (x, y, w, h) | None
    """
    cfg = config or Config()
    pred = predictor or _ensure_predictor(cfg.LEARNED_MODEL_PATH)

    # 0-255 正規化は前処理側で行われるため、そのまま渡す
    processed = preprocess_image(img, cfg)

    result = detect_landmarks(processed, pred, cfg, log_manager=None)

    # 整合性チェック
    if result.is_detected and len(result.landmarks_list) == 0:
        result.is_detected = False

    if result.is_detected:
        landmarks = result.landmarks_list[0]
    else:
        landmarks = cfg.TEMPLATE_LANDMARKS

    return landmarks.astype(np.int32), result.is_detected, result.bounding_box


def landmarks_from_path(
    img_path: str,
    config: Optional[Config] = None,
    predictor: Optional[dlib.shape_predictor] = None,
) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:
    """画像パスから68点ランドマークを返す。

    備考:
        既存の実装では .npy 画像を想定しているため、まずは .npy のロードを試み、
        失敗した場合は一般的な画像として読み取る。
    """
    cfg = config or Config()
    pred = predictor or _ensure_predictor(cfg.LEARNED_MODEL_PATH)

    img: Optional[np.ndarray] = None

    # .npy まず試す
    try:
        if img_path.lower().endswith(".npy"):
            img = np.load(img_path)
    except Exception:
        img = None

    # 画像として読み取り（OpenCV 非依存: 最低限の読み込み互換のため NumPy のみ）
    if img is None:
        try:
            import cv2  # 遅延import
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise RuntimeError(f"画像の読み込みに失敗しました: {img_path} - {e}")

    if img is None:
        raise RuntimeError(f"画像の読み込みに失敗しました: {img_path}")

    return landmarks_from_array(img, cfg, pred)


