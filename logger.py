"""ログ管理モジュール"""

import os


class LogManager:
    """ログ管理クラス"""
    
    def __init__(self, log_file: str = 'processed_data/error_log.txt'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_error(self, error_msg: str) -> None:
        """エラーメッセージをログファイルに記録"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{error_msg}\n")
            f.write("=" * 30 + "\n")
        print(error_msg)
