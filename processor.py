"""画像処理実行モジュール"""

import os
import cv2
import dlib
import numpy as np
from typing import Tuple, Optional

from config import Config
from data_models import ProcessResult
from logger import LogManager
from image_processor import preprocess_image
from landmark_detector import detect_landmarks
from file_handler import save_processed_files


def process_image(
    img_path: str,
    orignorm_dir: str,
    processed_dir: str,
    landmarks_dir: str,
    predictor: dlib.shape_predictor,
    config: Config,
    log_manager: Optional[LogManager] = None
) -> ProcessResult:
    """画像を処理してランドマークを検出する
    
    Args:
        img_path: 画像ファイルパス
        orignorm_dir: 正規化画像の保存先
        processed_dir: 処理済み画像の保存先
        landmarks_dir: ランドマークの保存先
        predictor: dlibのランドマーク予測器
        config: 設定オブジェクト
        log_manager: ログマネージャー（オプション）
        
    Returns:
        ProcessResult: 処理結果
    """
    try:
        # メモリ効率を改善するために、必要な部分だけを読み込む
        original_img = np.load(img_path, mmap_mode='r')
        original_img = np.array(original_img)
        
        # オリジナル画像を0-255に正規化
        orig_norm = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 前処理画像
        processed = preprocess_image(original_img, config)
        
        # メモリ解放
        del original_img
        
        # ランドマーク検出
        detection_result = detect_landmarks(processed, predictor, config, log_manager)
        
        # is_detectedの値とlandmarks_listの内容に整合性があることを確認
        if detection_result.is_detected and len(detection_result.landmarks_list) == 0:
            detection_result.is_detected = False
            detection_result.best_upsample = None
        
        # 比較画像の保存ディレクトリを設定
        comparison_dir = os.path.join(os.path.dirname(orignorm_dir), 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # ランドマークの決定と保存
        if detection_result.is_detected:
            landmarks = detection_result.landmarks_list[0]
            message = ""
        else:
            landmarks = config.TEMPLATE_LANDMARKS
            message = "顔が検出できませんでした"
        
        # ファイルの保存
        save_processed_files(
            img_path=img_path,
            orig_norm=orig_norm,
            processed=processed,
            landmarks=landmarks,
            orignorm_dir=orignorm_dir,
            processed_dir=processed_dir,
            landmarks_dir=landmarks_dir,
            comparison_dir=comparison_dir,
            is_detected=detection_result.is_detected
        )
        
        return ProcessResult(
            is_detected=detection_result.is_detected,
            message=message,
            best_upsample=detection_result.best_upsample,
            detection_info=detection_result.detection_info
        )
        
    except Exception as e:
        error_msg = f"処理エラー: {str(e)}"
        if log_manager:
            log_manager.log_error(error_msg)
        else:
            print(error_msg)
        return ProcessResult(
            is_detected=False,
            message=error_msg,
            best_upsample=None,
            detection_info=[]
        )


def process_image_wrapper(args: Tuple[str, str, str, str, Config]) -> ProcessResult:
    """マルチプロセス用のラッパー関数
    
    Args:
        args: (img_file, orignorm_dir, processed_dir, landmarks_dir, config)のタプル
        
    Returns:
        ProcessResult: 処理結果
    """
    try:
        img_file, orignorm_dir, processed_dir, landmarks_dir, config = args
        predictor = dlib.shape_predictor(config.LEARNED_MODEL_PATH)
        log_manager = LogManager()
        return process_image(
            img_file, orignorm_dir, processed_dir, landmarks_dir,
            predictor, config, log_manager
        )
    except Exception as e:
        error_msg = f"ラッパーエラー: {str(e)}"
        log_manager = LogManager()
        log_manager.log_error(error_msg)
        return ProcessResult(
            is_detected=False,
            message=error_msg,
            best_upsample=None,
            detection_info=[]
        )
