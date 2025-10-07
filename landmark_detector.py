"""ランドマーク検出モジュール"""

import dlib
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from constants import Constants
from data_models import DetectionInfo, DetectionResult

if TYPE_CHECKING:
    from config import Config
    from logger import LogManager


def detect_landmarks(
    processed_img: np.ndarray,
    predictor: dlib.shape_predictor,
    config: 'Config',
    log_manager: Optional['LogManager'] = None
) -> DetectionResult:
    """画像からランドマークを検出する
    
    Args:
        processed_img: 前処理済み画像
        predictor: dlibのランドマーク予測器
        config: 設定オブジェクト
        log_manager: ログマネージャー（オプション）
        
    Returns:
        DetectionResult: 検出結果
    """
    detector = dlib.get_frontal_face_detector()
    best_rects = []
    best_upsample = 0
    detection_info: List[DetectionInfo] = []
    
    # モードに応じてアップサンプリング回数を設定
    if config.DETECTION_MODE == 'high':
        upsample_times = [0, 1, 2, 3, 4]
    else:  # normal mode
        upsample_times = [0, 1, 2, 3]
    
    for upsample in upsample_times:
        try:
            rects = detector(processed_img, upsample)
            current_info = DetectionInfo(
                upsample=upsample,
                reason='顔が検出されませんでした'
            )
            
            if len(rects) > 0:
                # 最初に検出された顔を使用
                if not best_rects:
                    best_rects = [rects[0]]
                    best_upsample = upsample
                    current_info.reason = '成功'
            
        except Exception as e:
            current_info.reason = f'エラー: {str(e)}'
        
        detection_info.append(current_info)
        
        # 検出に成功したら終了
        if best_rects:
            break
    
    # 検出が成功したかどうかの判定
    is_detected = len(best_rects) > 0
    landmarks_list: List[np.ndarray] = []
    
    if is_detected:
        for rect in best_rects:
            try:
                shape = predictor(processed_img, rect)
                landmarks = np.array(
                    [[shape.part(i).x, shape.part(i).y] for i in range(Constants.NUM_LANDMARKS)]
                )
                landmarks_list.append(landmarks.astype(np.int32))
            except Exception as e:
                error_msg = f"ランドマーク処理エラー: {str(e)}"
                if log_manager:
                    log_manager.log_error(error_msg)
                else:
                    print(error_msg)
                continue
    
    if not is_detected:
        best_upsample = None
    
    return DetectionResult(
        landmarks_list=landmarks_list,
        best_upsample=best_upsample,
        detection_info=detection_info,
        is_detected=is_detected
    )
