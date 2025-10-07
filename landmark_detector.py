"""ランドマーク検出モジュール"""

import dlib
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

# 定数定義
NUM_LANDMARKS = 68
from data_types import DetectionInfo, DetectionResult

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
    best_bounding_box: Optional[Tuple[int, int, int, int]] = None
    adjusted_bounding_box: Optional[Tuple[int, int, int, int]] = None
    
    # モードに応じてアップサンプリング回数を設定
    if config.DETECTION_MODE == 'high':
        upsample_times = [1, 2]  # high mode: 1, 2回（必ずアップサンプリング）
    else:  # normal mode
        upsample_times = [0]  # normal mode: 0回のみ
    
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
                    # バウンディングボックス情報を保存（調整前の矩形）
                    rect = rects[0]
                    best_bounding_box = (rect.left(), rect.top(), rect.width(), rect.height())
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
                # 矩形のサイズを調整
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                center_x = x + w / 2
                center_y = y + h / 2
                new_w = w * config.BOUNDING_BOX_SCALE_X
                new_h = h * config.BOUNDING_BOX_SCALE_Y
                new_x = center_x - new_w / 2
                new_y = center_y - new_h / 2
                
                # 調整された矩形を作成
                adjusted_rect = dlib.rectangle(int(new_x), int(new_y), 
                                             int(new_x + new_w), int(new_y + new_h))
                
                # 調整された矩形でランドマーク検出
                shape = predictor(processed_img, adjusted_rect)
                landmarks = np.array(
                    [[shape.part(i).x, shape.part(i).y] for i in range(NUM_LANDMARKS)]
                )
                landmarks_list.append(landmarks.astype(np.int32))
                
                # 調整された矩形の情報を保存
                adjusted_bounding_box = (int(new_x), int(new_y), int(new_w), int(new_h))
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
        is_detected=is_detected,
        bounding_box=adjusted_bounding_box if adjusted_bounding_box else best_bounding_box
    )
