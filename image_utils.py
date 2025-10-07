"""画像処理ユーティリティモジュール"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def setup_directories(output_base_path: str, input_dir: str) -> Tuple[str, str, str]:
    """出力ディレクトリを設定する
    
    Args:
        output_base_path: 出力のベースパス
        input_dir: 入力ディレクトリパス
        
    Returns:
        (orignorm_dir, processed_dir, landmarks_dir)のタプル
    """
    dir_name = os.path.basename(input_dir)
    orignorm_dir = os.path.join(output_base_path, dir_name, 'orignorm')
    processed_dir = os.path.join(output_base_path, dir_name, 'processed')
    landmarks_dir = os.path.join(output_base_path, dir_name, 'landmarks')
    os.makedirs(orignorm_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(landmarks_dir, exist_ok=True)
    return orignorm_dir, processed_dir, landmarks_dir


def visualize_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    landmarks: List[np.ndarray],
    save_path: str,
    bounding_box: Optional[Tuple[int, int, int, int]] = None
) -> None:
    """オリジナル画像、処理済み画像、ランドマークを比較する画像を作成
    
    Args:
        original: オリジナル画像
        processed: 処理済み画像
        landmarks: ランドマークのリスト
        save_path: 保存先パス
        bounding_box: バウンディングボックス (x, y, width, height) - 既に調整済み
    """
    plt.figure(figsize=(15, 5))
    
    # オリジナル画像
    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    if bounding_box:
        x, y, w, h = bounding_box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)
    plt.title('Original Image')
    plt.axis('off')
    
    # 処理後画像
    plt.subplot(132)
    plt.imshow(processed, cmap='gray')
    if bounding_box:
        x, y, w, h = bounding_box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)
    plt.title('Processed Image')
    plt.axis('off')
    
    # ランドマーク付き画像
    plt.subplot(133)
    plt.imshow(processed, cmap='gray')
    if bounding_box:
        x, y, w, h = bounding_box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)
    if len(landmarks) > 0:
        plt.plot(landmarks[0][:, 0], landmarks[0][:, 1], 'r.', markersize=2)
    plt.title('Landmarks + Bounding Box')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_processed_files(
    img_path: str,
    orig_norm: np.ndarray,
    processed: np.ndarray,
    landmarks: np.ndarray,
    orignorm_dir: str,
    processed_dir: str,
    landmarks_dir: str,
    comparison_dir: str,
    is_detected: bool,
    bounding_box: Optional[Tuple[int, int, int, int]] = None
) -> None:
    """処理済みファイルを保存する
    
    Args:
        img_path: 画像ファイルパス
        orig_norm: 正規化済みオリジナル画像
        processed: 処理済み画像
        landmarks: ランドマーク
        orignorm_dir: オリジナル正規化画像の保存先
        processed_dir: 処理済み画像の保存先
        landmarks_dir: ランドマークの保存先
        comparison_dir: 比較画像の保存先
        is_detected: 検出成功フラグ
    """
    suffix = '' if is_detected else '_ng'
    base_name = os.path.basename(img_path).replace('.npy', '')
    
    # ファイルの保存
    np.save(
        os.path.join(orignorm_dir, f'{base_name}_orignorm{suffix}.npy'),
        orig_norm
    )
    np.save(
        os.path.join(processed_dir, f'{base_name}_processed{suffix}.npy'),
        processed
    )
    np.save(
        os.path.join(landmarks_dir, f'{base_name}_landmarks{suffix}.npy'),
        landmarks
    )
    
    # 比較画像の保存
    comparison_path = os.path.join(
        comparison_dir,
        f'{base_name}_comparison{suffix}.png'
    )
    visualize_comparison(orig_norm, processed, [landmarks], comparison_path, bounding_box)
