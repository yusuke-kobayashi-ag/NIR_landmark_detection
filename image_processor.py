"""画像処理モジュール"""

import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config


def preprocess_image(img: np.ndarray, config: 'Config') -> np.ndarray:
    """画像の前処理を行う
    
    Args:
        img: 入力画像
        config: 設定オブジェクト
        
    Returns:
        前処理済み画像
    """
    params = config.IMAGE_PROCESSING
    processed = img.copy()
    if len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # 画像をuint8に変換
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ガンマ補正用のLUTを作成
    gamma = params['gamma']
    lookUpTable = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        lookUpTable[i, 0] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    
    # LUTを適用
    processed = cv2.LUT(processed, lookUpTable)
    
    processed = cv2.bilateralFilter(
        processed,
        params['bilateral_d'],
        params['bilateral_sigma_color'],
        params['bilateral_sigma_space']
    )
    clahe = cv2.createCLAHE(clipLimit=params['contrast_clip'], tileGridSize=(16,16))
    processed = clahe.apply(processed)
    processed = cv2.convertScaleAbs(processed, alpha=params['alpha'], beta=params['beta'])
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    return processed
