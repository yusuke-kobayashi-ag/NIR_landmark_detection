"""NIR画像の顔ランドマーク検出メインスクリプト"""

import os
import argparse
import tkinter as tk
from create_folder_list import FolderListCreator
from directory_processor import process_directory


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='NIR画像の処理と顔ランドマーク検出')
    parser.add_argument(
        '--dirs',
        nargs='+',
        help='処理するNIR画像が含まれるディレクトリのパス（複数指定可能）'
    )
    parser.add_argument(
        '--list',
        help='処理するディレクトリのパスが記載されたテキストファイルのパス'
    )
    parser.add_argument(
        '--mode',
        choices=['normal', 'high'],
        default='normal',
        help='検出モード: normal (3回まで) または high (4回まで)'
    )
    parser.add_argument(
        '--create-list',
        action='store_true',
        help='フォルダリスト作成ツールを起動し、処理を開始'
    )
    args = parser.parse_args()
    
    # フォルダリスト作成モード
    if args.create_list:
        root = tk.Tk()
        app = FolderListCreator(root)
        root.mainloop()
        # GUIが閉じられたら終了
        exit(0)
    
    # 処理対象のディレクトリリストを取得
    if args.list:
        try:
            with open(args.list, 'r', encoding='utf-8') as f:
                input_dirs = [line.strip() for line in f if line.strip()]
            if not input_dirs:
                print("エラー: テキストファイルに有効なディレクトリパスが含まれていません。")
                exit(1)
        except Exception as e:
            print(f"エラー: テキストファイルの読み込みに失敗しました: {str(e)}")
            exit(1)
    elif args.dirs:
        input_dirs = args.dirs
    else:
        # デフォルトでカレントディレクトリを使用
        input_dirs = ['.']
    
    # 各ディレクトリを順番に処理
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"警告: {input_dir} は有効なディレクトリではありません。スキップします。")
            continue
        print(f"\n=== ディレクトリ {input_dir} の処理を開始します ===")
        process_directory(input_dir, args.mode)
        print(f"=== ディレクトリ {input_dir} の処理が完了しました ===\n")


if __name__ == "__main__":
    main()