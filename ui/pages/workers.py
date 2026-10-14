"""图像预处理后台线程与进度对话框封装。"""
import os
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressDialog


class PreprocessThread(QThread):
    """批量图像预处理线程（去雾/增强等），处理结果与对比图写入输出目录。"""
    progress = pyqtSignal(int)          # 进度 0~100
    finished = pyqtSignal(str, str)     # (功能名, 输出目录)

    def __init__(self, files, output_dir, func, func_name, **kwargs):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.func = func
        self.func_name = func_name
        self.kwargs = kwargs

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)

        n_files = len(self.files)
        for i, path in enumerate(self.files, start=1):
            try:
                data_hazy, clean_image = self.func(path, **self.kwargs)
                save_path = os.path.join(self.output_dir, f'{self.func_name}_{Path(path).name}')
                cv2.imwrite(save_path, clean_image)
                print(f'{i} save {self.func_name} image to {save_path}')

                save_path = os.path.join(self.output_dir, f'combined_{Path(path).name}')
                combined = np.hstack((data_hazy, clean_image))
                cv2.imwrite(save_path, combined)

                progress = int(i / n_files * 100)
                self.progress.emit(progress)
            except Exception as e:
                print(e)
                break

        self.finished.emit(self.func_name, self.output_dir)


def run_preprocess(parent, files, output_dir, func, func_name, on_finished=None, **kwargs):
    """启动一个预处理线程并附带模态进度对话框。

    返回 worker 线程对象（调用方需保留引用，如 ``self._worker``）。
    on_finished: 签名为 func_name, output_dir -> None 的回调。
    """
    progress_dialog = QProgressDialog("正在处理……", "取消", 0, 100, parent)
    progress_dialog.setWindowTitle("任务进度")
    progress_dialog.setAutoClose(True)   # 100% 自动关闭
    progress_dialog.setAutoReset(True)
    progress_dialog.setModal(True)

    worker = PreprocessThread(files, output_dir, func, func_name, **kwargs)
    worker.progress.connect(progress_dialog.setValue)
    if on_finished is not None:
        worker.finished.connect(on_finished)
    worker._progress_dialog = progress_dialog  # 防止对话框被 GC
    worker.start()
    return worker
