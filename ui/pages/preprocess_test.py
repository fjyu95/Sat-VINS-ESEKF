"""预处理算法测试：独立评估去雾/增强算法效果，与主流程数据隔离。

- 输出写入 preprocessing_test/<算法>/<时间戳>/，不修改 ctx.dehazed_dir/enhanced_dir
- 默认使用 preprocessing/*/test_data 下的测试数据，可选择新数据（目录或文件）
- 完成后弹出原图/结果并排对比窗口，附带信息熵、平均梯度、对比度、
  PSNR、SSIM、耗时等指标，汇总指标写入 metrics.csv
"""
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGraphicsPixmapItem,
    QGraphicsScene, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressDialog, QPushButton, QVBoxLayout, QWidget,
)

from preprocessing.dehaze.dehaze import dehaze_image
from preprocessing.enhance.enhance import clahe_enhance
from preprocessing.metrics import METRIC_FIELDS, compute_metrics, format_metrics
from stats_analysis.pandas_viewer import PandasViewer
from ui.pages.common import ZoomableGraphicsView
from ui.pages.context import IMAGE_EXTS

TEST_OUTPUT_ROOT = Path('preprocessing_test')

# 与主流程一致的固定算法：去雾用 AOD-Net，增强用 CLAHE
ALGOS = [
    dict(label='图像去雾', key='dehazed', func=dehaze_image,
         data_dir='preprocessing/dehaze/test_data', kwargs={}),
    dict(label='图像增强', key='enhanced', func=clahe_enhance,
         data_dir='preprocessing/enhance/test_data', kwargs={}),
]


def _ensure_u8(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    if img.size and img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _match_channels(a, b):
    """统一通道数，便于 hstack 拼接对比图。"""
    if a.ndim == 2 and b.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    elif a.ndim == 3 and b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    return a, b


# ----------------------------------------------------------------------
# 后台测试线程
# ----------------------------------------------------------------------
class AlgoTestThread(QThread):
    """算法测试线程：逐图运行算法、计算指标并写入独立输出目录。"""
    progress = pyqtSignal(int)           # 进度 0~100
    finished = pyqtSignal(list, str)     # (records, output_dir)

    def __init__(self, files, func, output_dir, **kwargs):
        super().__init__()
        self.files = files
        self.func = func
        self.output_dir = output_dir
        self.kwargs = kwargs

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        records = []
        n_files = len(self.files)
        for i, path in enumerate(self.files, start=1):
            name = Path(path).name
            try:
                t0 = time.perf_counter()
                before, after = self.func(path, **self.kwargs)
                elapsed = time.perf_counter() - t0

                before, after = _match_channels(_ensure_u8(before), _ensure_u8(after))
                dst = os.path.join(self.output_dir, f'result_{name}')
                cv2.imwrite(dst, after)
                cv2.imwrite(os.path.join(self.output_dir, f'combined_{name}'),
                            np.hstack((before, after)))

                m = compute_metrics(before, after)
                m['time'] = elapsed
                records.append({'name': name, 'src': str(path), 'dst': dst, 'metrics': m})
                print(f'{i}/{n_files} tested: {name}')
            except Exception as e:
                print(f'{i}/{n_files} failed: {name}: {e}')
            self.progress.emit(int(i / n_files * 100))

        self._save_csv(records)
        self.finished.emit(records, self.output_dir)

    def _save_csv(self, records):
        if not records:
            return
        csv_path = os.path.join(self.output_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(METRIC_FIELDS)
            for r in records:
                m = r['metrics']
                writer.writerow([r['name'],
                                 m['entropy_before'], m['entropy_after'],
                                 m['avg_grad_before'], m['avg_grad_after'],
                                 m['contrast_before'], m['contrast_after'],
                                 m['psnr'], m['ssim'], m['time']])


# ----------------------------------------------------------------------
# 结果对比查看器
# ----------------------------------------------------------------------
class CompareViewer(QMainWindow):
    """原图/结果并排对比浏览器（滚轮缩放 + 拖拽平移），附指标文本。"""

    def __init__(self, algo_label, records, output_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'算法测试对比 - {algo_label}')
        self.resize(1100, 680)
        self.records = records
        self.output_dir = output_dir
        self.index = 0
        self._viewers = []

        central = QWidget()
        root = QVBoxLayout(central)

        panes = QHBoxLayout()
        pane_src, self.view_src, self.scene_src = self._make_pane('原图')
        pane_dst, self.view_dst, self.scene_dst = self._make_pane('处理结果')
        panes.addWidget(pane_src, 1)
        panes.addWidget(pane_dst, 1)
        root.addLayout(panes, 1)

        self.info_label = QLabel()
        self.info_label.setStyleSheet('font-family: Consolas, monospace;')
        root.addWidget(self.info_label)

        nav = QHBoxLayout()
        btn_prev = QPushButton('上一张')
        btn_next = QPushButton('下一张')
        btn_stats = QPushButton('指标统计表')
        btn_dir = QPushButton('打开输出目录')
        btn_prev.clicked.connect(self.show_prev)
        btn_next.clicked.connect(self.show_next)
        btn_stats.clicked.connect(self.show_metrics_table)
        btn_dir.clicked.connect(self.open_output_dir)
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(btn_prev)
        nav.addWidget(self.page_label, 1)
        nav.addWidget(btn_next)
        nav.addWidget(btn_stats)
        nav.addWidget(btn_dir)
        root.addLayout(nav)
        self.setCentralWidget(central)

        self._show_average()
        self._load(0)

    @staticmethod
    def _make_pane(title):
        scene = QGraphicsScene()
        view = ZoomableGraphicsView()
        view.setScene(scene)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        cap = QLabel(title)
        cap.setAlignment(Qt.AlignCenter)
        v.addWidget(cap)
        v.addWidget(view, 1)
        return w, view, scene

    def _load(self, idx):
        if not self.records:
            return
        self.index = idx % len(self.records)
        rec = self.records[self.index]
        for scene, view, path in ((self.scene_src, self.view_src, rec['src']),
                                  (self.scene_dst, self.view_dst, rec['dst'])):
            scene.clear()
            pix = QPixmap(path)
            if pix.isNull():
                continue
            item = QGraphicsPixmapItem(pix)
            scene.addItem(item)
            scene.setSceneRect(item.boundingRect())
            view.fitInView(item, Qt.KeepAspectRatio)
        self.info_label.setText(f"{rec['name']}\n{format_metrics(rec['metrics'])}")
        self.page_label.setText(f"{self.index + 1} / {len(self.records)}")

    def _show_average(self):
        """状态栏显示整批图像的均值指标。"""
        if not self.records:
            return
        keys = ['entropy_after', 'avg_grad_after', 'contrast_after', 'time']
        avg = {k: np.nanmean([r['metrics'][k] for r in self.records]) for k in keys}
        self.statusBar().showMessage(
            f"共 {len(self.records)} 张 | 均值: "
            f"熵 {avg['entropy_after']:.3f} bit | "
            f"平均梯度 {avg['avg_grad_after']:.3f} | "
            f"对比度σ {avg['contrast_after']:.2f} | "
            f"耗时 {avg['time']:.2f} s/张")

    def show_prev(self):
        self._load(self.index - 1)

    def show_next(self):
        self._load(self.index + 1)

    def show_metrics_table(self):
        """以表格形式打开本批全部指标（可排序）。"""
        df = pd.DataFrame([dict(file=r['name'], **r['metrics']) for r in self.records])
        df = df.rename(columns={'time': 'time_sec'})[METRIC_FIELDS]
        viewer = PandasViewer(self, df=df)
        viewer.setWindowTitle('算法测试指标统计')
        self._viewers.append(viewer)
        viewer.show()

    def open_output_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.output_dir)))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_PageUp):
            self.show_prev()
        elif event.key() in (Qt.Key_Right, Qt.Key_PageDown, Qt.Key_Space):
            self.show_next()
        else:
            super().keyPressEvent(event)


# ----------------------------------------------------------------------
# 算法测试对话框
# ----------------------------------------------------------------------
class AlgoTestDialog(QDialog):
    """预处理算法测试对话框：选择算法与测试数据，后台批处理并弹对比窗口。"""

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle('预处理算法测试')
        self.resize(560, 240)
        self._worker = None
        self._viewers = []          # 防止对比窗口被 GC
        self._picked_files = None   # 「选择文件」选中的文件列表

        form = QFormLayout()

        self.algo_combo = QComboBox()
        for a in ALGOS:
            self.algo_combo.addItem(a['label'])
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        form.addRow('测试算法', self.algo_combo)

        src_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.textChanged.connect(lambda *_: setattr(self, '_picked_files', None))
        btn_dir = QPushButton('选择目录…')
        btn_files = QPushButton('选择文件…')
        btn_dir.clicked.connect(self._choose_dir)
        btn_files.clicked.connect(self._choose_files)
        src_row.addWidget(self.path_edit, 1)
        src_row.addWidget(btn_dir)
        src_row.addWidget(btn_files)
        form.addRow('测试数据', src_row)

        hint = QLabel('默认使用内置测试数据，可自选目录或文件；'
                      '结果输出至 preprocessing_test/，不影响主流程。')
        hint.setWordWrap(True)
        form.addRow('', hint)

        btn_run = QPushButton('开始测试')
        btn_run.setMinimumHeight(32)
        btn_run.clicked.connect(self.run_test)
        form.addRow('', btn_run)

        self.status_label = QLabel('状态：待机')
        form.addRow('', self.status_label)
        self.setLayout(form)

        self._on_algo_changed(0)

    # ------------------------------------------------------------------
    # 数据源选择
    # ------------------------------------------------------------------
    def _on_algo_changed(self, idx):
        self._picked_files = None
        self.path_edit.setText(ALGOS[idx]['data_dir'])

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, '选择测试数据目录',
                                             self.path_edit.text() or '.')
        if d:
            self.path_edit.setText(d)

    def _choose_files(self):
        exts = ' '.join(f'*{e}' for e in IMAGE_EXTS)
        files, _ = QFileDialog.getOpenFileNames(
            self, '选择测试图像', self.path_edit.text() or '.', f'Images ({exts})')
        if files:
            self.path_edit.setText(f'<已选择 {len(files)} 个文件>')
            self._picked_files = files

    def _collect_files(self):
        if self._picked_files:
            return list(self._picked_files)
        text = self.path_edit.text().strip()
        if not text or text.startswith('<'):
            return []
        p = Path(text)
        if p.is_dir():
            return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return [str(p)]
        return []

    # ------------------------------------------------------------------
    # 运行
    # ------------------------------------------------------------------
    def _set_status(self, text):
        self.status_label.setText(f'状态：{text}')
        print(f'>>> {text}')

    def run_test(self):
        algo = ALGOS[self.algo_combo.currentIndex()]
        files = self._collect_files()
        if not files:
            QMessageBox.warning(self, '提示', '未找到可测试的图像文件，请检查测试数据路径。',
                                QMessageBox.Ok)
            return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = (TEST_OUTPUT_ROOT / algo['key'] / stamp).as_posix()

        kwargs = dict(algo['kwargs'])
        if algo['key'] == 'enhanced' and self.ctx.config:
            kwargs['clip_limit'] = self.ctx.config.get('clahe_clip_limit', 2.0)

        self._set_status(f"{algo['label']} 测试中……（{len(files)} 张）")
        progress = QProgressDialog('正在处理……', '取消', 0, 100, self)
        progress.setWindowTitle('任务进度')
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setModal(True)

        worker = AlgoTestThread(files, algo['func'], output_dir, **kwargs)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(
            lambda records, outdir: self._on_finished(algo, records, outdir))
        worker._progress_dialog = progress   # 防止对话框被 GC
        worker.start()
        self._worker = worker

    def _on_finished(self, algo, records, output_dir):
        if not records:
            self._set_status('测试失败：所有图像均未处理成功')
            QMessageBox.warning(self, '提示',
                                '所有图像均未处理成功，详见控制台输出。', QMessageBox.Ok)
            return
        self._set_status(f"测试完成：{len(records)} 张 → {output_dir}")

        viewer = CompareViewer(algo['label'], records, output_dir, self)
        self._viewers.append(viewer)
        viewer.show()
