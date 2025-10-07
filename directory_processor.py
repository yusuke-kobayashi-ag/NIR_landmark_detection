"""ディレクトリ処理モジュール"""

import os
import glob
import numpy as np
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

from config import Config
from logger import LogManager
from data_types import DetectionInfo
from image_utils import setup_directories, visualize_comparison
from processor import process_image_wrapper


def process_directory(input_dir: str, detection_mode: str = 'normal') -> None:
    """ディレクトリ内のすべての画像を処理する
    
    Args:
        input_dir: 入力ディレクトリパス
        detection_mode: 検出モード ('normal' または 'high')
    """
    config = Config()
    config.DETECTION_MODE = detection_mode
    log_manager = LogManager()
    
    not_detected: List[Tuple[str, str, List[DetectionInfo]]] = []
    detection_results: List[Tuple[str, Optional[int], bool]] = []
    last_successful_landmarks: Optional[np.ndarray] = None
    
    # 入力ディレクトリ内の.npyファイルを取得
    img_files = glob.glob(os.path.join(input_dir, '*.npy'))
    if not img_files:
        print(f"エラー: {input_dir} 内に.npyファイルが見つかりません。")
        return
    
    # 出力ディレクトリの設定
    orignorm_dir, processed_dir, landmarks_dir = setup_directories(
        config.OUTPUT_BASE_DIR, input_dir
    )
    
    # CPUコア数を取得
    if multiprocessing.cpu_count() > 2:
        max_workers = multiprocessing.cpu_count() - 2
    else:
        max_workers = 1
    
    # 画像処理の実行
    args_list = [
        (img_file, orignorm_dir, processed_dir, landmarks_dir, config)
        for img_file in img_files
    ]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_image_wrapper, args): args[0]
            for args in args_list
        }
        
        # 進捗バーの設定
        pbar = tqdm(total=len(future_to_file), desc="画像処理中", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        success_count = 0
        failure_count = 0
        
        for future in as_completed(future_to_file):
            img_file = future_to_file[future]
            base_filename = os.path.basename(img_file).replace('.npy', '')
            
            try:
                result = future.result()
                
                if result.is_detected:
                    # 成功時
                    success_count += 1
                    tqdm.write(f"✅ 成功: {base_filename}")
                    # 成功したランドマークを保存
                    landmarks_path = os.path.join(
                        landmarks_dir, f"{base_filename}_landmarks.npy"
                    )
                    if os.path.exists(landmarks_path):
                        last_successful_landmarks = np.load(landmarks_path)
                else:
                    # 失敗時
                    failure_count += 1
                    tqdm.write(f"❌ 失敗: {base_filename} - {result.message}")
                    not_detected.append((base_filename, result.message, result.detection_info))
                    
                    # 直前の成功したランドマークがある場合はそれを使用
                    if last_successful_landmarks is not None:
                        landmarks_path = os.path.join(
                            landmarks_dir, f"{base_filename}_landmarks_ng.npy"
                        )
                        np.save(landmarks_path, last_successful_landmarks)
                        # 比較画像も更新
                        orig_norm = np.load(
                            os.path.join(orignorm_dir, f"{base_filename}_orignorm_ng.npy")
                        )
                        processed = np.load(
                            os.path.join(processed_dir, f"{base_filename}_processed_ng.npy")
                        )
                        comparison_dir = os.path.join(os.path.dirname(orignorm_dir), 'comparisons')
                        comparison_path = os.path.join(
                            comparison_dir, f"{base_filename}_comparison_ng.png"
                        )
                        visualize_comparison(
                            orig_norm, processed, [last_successful_landmarks], comparison_path, None
                        )
                
                detection_results.append((
                    base_filename,
                    result.best_upsample,
                    result.is_detected
                ))
                
            except Exception as e:
                # 処理自体の例外
                failure_count += 1
                error_msg = f"処理例外: {str(e)}"
                tqdm.write(f"❌ エラー: {base_filename} - {error_msg}")
                log_manager.log_error(error_msg)
                not_detected.append((base_filename, error_msg, []))
                detection_results.append((base_filename, None, False))
            
            # 進捗バーを更新
            pbar.update(1)
            pbar.set_postfix({
                '成功': success_count, 
                '失敗': failure_count,
                '成功率': f"{success_count/(success_count+failure_count)*100:.1f}%" if (success_count+failure_count) > 0 else "0%"
            })
        
        # 進捗バーを閉じる
        pbar.close()
    
    # 検出結果をファイルに保存
    result_file = os.path.join(config.OUTPUT_BASE_DIR, 'detection_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("ファイル名,最適なパラメータ,検出結果\n")
        for base_name, best_upsample, success in detection_results:
            result_filename = f"{base_name}{'_ng' if not success else ''}.npy"
            if best_upsample is not None:
                f.write(f"{result_filename},upsample:{best_upsample}")
            else:
                f.write(f"{result_filename},検出失敗")
            f.write(f",{'成功' if success else '失敗'}\n")
    
    # 検出失敗の結果をファイルに保存
    if not_detected:
        out_txt = os.path.join(os.path.dirname(orignorm_dir), 'not_detected.txt')
        with open(out_txt, 'w', encoding='utf-8') as f:
            for base_name, message, detection_info in not_detected:
                f.write(f"{base_name}_ng.npy - {message}\n")
        print(f"\n顔が検出できなかったファイル一覧を {out_txt} に保存しました（{len(not_detected)}件）")
    else:
        print("\nすべての画像で顔が検出されました！")
    
    # 最終結果の表示
    total_processed = success_count + failure_count
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"🎯 処理完了サマリー")
    print(f"{'='*60}")
    print(f"📊 処理統計:")
    print(f"   • 総処理数: {total_processed} ファイル")
    print(f"   • 成功: {success_count} ファイル")
    print(f"   • 失敗: {failure_count} ファイル")
    print(f"   • 成功率: {success_rate:.1f}%")
    print(f"")
    print(f"📁 出力ディレクトリ:")
    print(f"   • オリジナル正規化画像: {orignorm_dir}")
    print(f"   • 処理済み画像: {processed_dir}")
    print(f"   • ランドマーク: {landmarks_dir}")
    print(f"   • 比較画像: {os.path.join(os.path.dirname(orignorm_dir), 'comparisons')}")
    print(f"   • 検出結果: {result_file}")
    print(f"{'='*60}")
