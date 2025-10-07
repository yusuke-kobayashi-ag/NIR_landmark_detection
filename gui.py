import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
import sys

class FolderListCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("NIR画像ランドマーク検出ツール")
        self.root.geometry("900x650")
        
        # フォルダリストを保持する変数
        self.folders = []
        
        # GUIの作成
        self.create_widgets()
        
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 設定フレーム
        settings_frame = ttk.LabelFrame(main_frame, text="設定", padding="5")
        settings_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # 検出モード選択
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(mode_frame, text="検出モード:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="normal")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, 
                                 values=["normal", "high"], state="readonly", width=10)
        mode_combo.pack(side=tk.LEFT, padx=5)
        
        # モード説明ラベル
        mode_info = ttk.Label(mode_frame, text="(normal: 高速, high: 高精度)", 
                             font=("TkDefaultFont", 8), foreground="gray")
        mode_info.pack(side=tk.LEFT, padx=5)
        
        # フィルター設定
        filter_frame = ttk.Frame(settings_frame)
        filter_frame.pack(side=tk.LEFT, padx=20, pady=5)
        ttk.Label(filter_frame, text="フォルダ名フィルター:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=50)
        filter_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(filter_frame, text="（カンマ区切りで複数指定可能）").pack(side=tk.LEFT, padx=5)
        
        # ボタンフレーム
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # フォルダ追加ボタン
        add_button = ttk.Button(button_frame, text="フォルダを追加", command=self.add_folder)
        add_button.pack(side=tk.LEFT, padx=5)
        
        # フォルダ一括追加ボタン
        add_multiple_button = ttk.Button(button_frame, text="フォルダを一括追加", command=self.add_multiple_folders)
        add_multiple_button.pack(side=tk.LEFT, padx=5)
        
        # フォルダ削除ボタン
        remove_button = ttk.Button(button_frame, text="選択したフォルダを削除", command=self.remove_folder)
        remove_button.pack(side=tk.LEFT, padx=5)
        
        # リストボックス
        self.listbox = tk.Listbox(main_frame, width=80, height=20)
        self.listbox.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.grid(row=2, column=2, sticky=(tk.N, tk.S))
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        # 保存ボタン
        save_button = ttk.Button(main_frame, text="リストを保存して処理開始", command=self.save_list)
        save_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # ステータスバー
        self.status_var = tk.StringVar()
        self.status_var.set("準備完了")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=tk.W)
        
    def add_folder(self):
        folder = filedialog.askdirectory(title="処理するフォルダを選択")
        if folder:
            if folder in self.folders:
                messagebox.showwarning("警告", "このフォルダは既にリストに追加されています。")
                return
            self.folders.append(folder)
            self.listbox.insert(tk.END, folder)
            self.status_var.set(f"フォルダを追加しました: {folder}")
    
    def add_multiple_folders(self):
        # 親フォルダの選択
        folders = filedialog.askdirectory(title="処理するフォルダを含む親フォルダを選択")
        if not folders:
            return
            
        # フィルター文字列を取得
        filter_text = self.filter_var.get().strip()
        filters = [f.strip() for f in filter_text.split(',')] if filter_text else []
            
        # 選択されたフォルダ内の.npyファイルを含むサブフォルダを探す
        found_folders = set()
        for root, dirs, files in os.walk(folders):
            if any(file.endswith('.npy') for file in files):
                # フィルター条件をチェック
                folder_name = os.path.basename(root)
                if not filters or any(f in folder_name for f in filters):
                    found_folders.add(root)
        
        if not found_folders:
            messagebox.showwarning("警告", "条件に一致するフォルダが見つかりませんでした。")
            return
            
        # 見つかったフォルダを追加
        added_count = 0
        for folder in sorted(found_folders):
            if folder not in self.folders:
                self.folders.append(folder)
                self.listbox.insert(tk.END, folder)
                added_count += 1
        
        self.status_var.set(f"{added_count}個のフォルダを追加しました")
        if added_count > 0:
            messagebox.showinfo("成功", f"{added_count}個のフォルダを追加しました。")
    
    def remove_folder(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "削除するフォルダを選択してください。")
            return
        index = selection[0]
        folder = self.folders.pop(index)
        self.listbox.delete(index)
        self.status_var.set(f"フォルダを削除しました: {folder}")
    
    def save_list(self):
        if not self.folders:
            messagebox.showwarning("警告", "保存するフォルダがありません。")
            return
        
        try:
            file_path = os.path.join('.', 'folder_list.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                for folder in self.folders:
                    f.write(f"{folder}\n")
            self.status_var.set(f"フォルダリストを保存しました: {file_path}")
            
            # メインプログラムを実行
            try:
                # メインプログラムのパスを取得
                main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
                
                # GUIを閉じる
                self.root.destroy()
                
                # メインプログラムを実行
                selected_mode = self.mode_var.get()
                subprocess.run([sys.executable, main_script, '--list', file_path, '--mode', selected_mode], check=True)
                
            except subprocess.CalledProcessError as e:
                messagebox.showerror("エラー", f"処理の実行中にエラーが発生しました: {str(e)}")
            except Exception as e:
                messagebox.showerror("エラー", f"予期せぬエラーが発生しました: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("エラー", f"ファイルの保存に失敗しました: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FolderListCreator(root)
    root.mainloop()
