"""データモデル定義モジュール"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DetectionInfo:
    """検出情報"""
    upsample: int
    reason: str


@dataclass
class DetectionResult:
    """ランドマーク検出の結果"""
    landmarks_list: List[np.ndarray]
    best_upsample: Optional[int]
    detection_info: List[DetectionInfo]
    is_detected: bool


@dataclass
class ProcessResult:
    """画像処理の結果"""
    is_detected: bool
    message: str
    best_upsample: Optional[int]
    detection_info: List[DetectionInfo]
