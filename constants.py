"""定数定義モジュール"""

from enum import Enum


class Constants:
    """処理で使用する定数"""
    # ランドマークインデックス
    NUM_LANDMARKS = 68


class DetectionMode(Enum):
    """検出モード"""
    NORMAL = 'normal'
    HIGH = 'high'
