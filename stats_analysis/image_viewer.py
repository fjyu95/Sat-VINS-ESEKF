# image_viewer.py

import os
import re
from math import isfinite

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QToolBar, QAction, QLabel, QMessageBox
)
from PyQt5.QtGui import QPixmap, QTransform, QKeySequence, QPainter
from PyQt5.QtCore import Qt, QSize, QTimer

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', ".webp", '.tif', '.tiff']


class ImageView(QGraphicsView):
    """GraphicsView with Ctrl+Wheel Zoom & Hand-Drag. Robust against zero/inf scales."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._scale = 1.0

    def wheelEvent(self, event):
        # Ctrl + wheel -> zoom, otherwise default (scroll)
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom_to(self, scale):
        """Set absolute scale factor (1.0 = actual size). Ignore invalid/zero scales."""
        try:
            if scale is None:
                return
            # protect against NaN/inf/<=0
            if not isfinite(scale) or scale <= 0:
                return
            # apply transform
            self.resetTransform()
            self.scale(scale, scale)
            self._scale = float(scale)
        except Exception:
            # on any unexpected error, fallback to 1.0
            self.resetTransform()
            self._scale = 1.0

    def zoom_in(self, factor=1.25):
        if factor <= 1.0:
            factor = 1.25
        cur = self._scale if (isfinite(self._scale) and self._scale > 0) else 1.0
        self.zoom_to(cur * factor)

    def zoom_out(self, factor=1.25):
        if factor <= 1.0:
            factor = 1.25
        cur = self._scale if (isfinite(self._scale) and self._scale > 0) else 1.0
        new = cur / factor
        # avoid tiny scales, clamp to a reasonable min (e.g. 0.01)
        if not isfinite(new) or new <= 0:
            new = 1.0
        self.zoom_to(new)

    def get_scale(self):
        return float(self._scale)


class ImageBrowser(QMainWindow):
    """独立图像浏览器窗口，可直接嵌入你的项目 (fixed issues)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Browser")
        self.resize(900, 650)

        # Scene & View
        self.scene = QGraphicsScene(self)
        self.view = ImageView(self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.pixmap_item = None
        self.image_files = []
        self.index = -1
        self.current_rot = 0

        # Toolbar
        self.init_toolbar()

        # Status bar
        self.label_name = QLabel("")
        self.label_zoom = QLabel("")
        self.statusBar().addWidget(self.label_name, 1)
        self.statusBar().addPermanentWidget(self.label_zoom)

    def init_toolbar(self):
        tb = QToolBar("Toolbar")
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        open_act = QAction("Open Folder", self)
        open_act.triggered.connect(self.open_folder_dialog)
        open_act.setShortcut(QKeySequence.Open)
        tb.addAction(open_act)

        tb.addSeparator()

        prev_act = QAction("Prev", self)
        prev_act.triggered.connect(self.show_prev)
        # 快捷键： Left / PageUp
        prev_act.setShortcut(QKeySequence(Qt.Key_Left))
        tb.addAction(prev_act)

        next_act = QAction("Next", self)
        next_act.triggered.connect(self.show_next)
        # 快捷键： Right / PageDown
        next_act.setShortcut(QKeySequence(Qt.Key_Right))
        tb.addAction(next_act)

        tb.addSeparator()

        z_in = QAction("Zoom In", self)
        z_in.triggered.connect(lambda: (self.view.zoom_in(), self.update_status()))
        z_in.setShortcut(QKeySequence.ZoomIn)
        tb.addAction(z_in)

        z_out = QAction("Zoom Out", self)
        z_out.triggered.connect(lambda: (self.view.zoom_out(), self.update_status()))
        z_out.setShortcut(QKeySequence.ZoomOut)
        tb.addAction(z_out)

        fit = QAction("Fit Window", self)
        fit.triggered.connect(self.fit_window)
        fit.setShortcut(QKeySequence("F"))
        tb.addAction(fit)

        actual = QAction("Actual Size", self)
        actual.triggered.connect(self.actual_size)
        actual.setShortcut(QKeySequence("A"))
        tb.addAction(actual)

        tb.addSeparator()

        rot_l = QAction("Rotate -90", self)
        rot_l.triggered.connect(lambda: (self.rotate(-90), self.update_status()))
        tb.addAction(rot_l)

        rot_r = QAction("Rotate +90", self)
        rot_r.triggered.connect(lambda: (self.rotate(90), self.update_status()))
        tb.addAction(rot_r)

    # ---------------- Folder Operations ----------------

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.open_folder(folder)
            # self.open_files(folder)

    def open_folder(self, folder):
        """外部可直接调用，传入目录打开"""
        files = sorted(os.listdir(folder))
        images = [
            os.path.join(folder, f)
            for f in files
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        if not images:
            QMessageBox.information(self, "No Images", "Folder contains no images.")
            return

        self.image_files = images
        self.index = 0
        self.load_image(0)

    def open_files(self, input_path, pattern=r".*\.(png|jpg|jpeg|bmp|tif|tiff|webp)$"):
        """
        支持三种调用方式：
        1. open_files("/path/to/dir")                     → 打开目录下所有图片
        2. open_files("/path/to/dir", r"IMG_\d+\.jpg")    → 正则过滤
        3. open_files(["1.jpg", "2.jpg", "3.png"])        → 直接传入文件列表

        参数:
            input_path : str 或 list[str]
            pattern    : 正则过滤（对目录生效）
        """

        # 直接传入文件列表
        if isinstance(input_path, list):
            files = [f for f in input_path if os.path.isfile(f)]
            if not files:
                QMessageBox.warning(self, "错误", "传入的文件列表无有效图像！")
                return

            self.images = files
            self.current_index = 0
            self.load_image(0)
            return

        # 传入的是目录
        if isinstance(input_path, str) and os.path.isdir(input_path):
            try:
                reg = re.compile(pattern, re.IGNORECASE)
            except Exception as e:
                QMessageBox.warning(self, "错误", f"正则表达式错误: {e}")
                return

            all_files = os.listdir(input_path)
            files = [
                os.path.join(input_path, f)
                for f in all_files
                if reg.match(f)  # 正则匹配文件名
            ]

            if not files:
                QMessageBox.information(self, "提示", "目录中没有匹配的图像文件")
                return

            self.images = sorted(files)
            self.current_index = 0
            self.load_image(0)
            return

        # 传入的是单独文件
        if isinstance(input_path, str) and os.path.isfile(input_path):
            self.images = [input_path]
            self.current_index = 0
            self.load_image(0)
            return

        # 其他异常情况
        QMessageBox.warning(self, "错误", "open_files() 的输入无效！")

    # ---------------- Image Operations ----------------

    def load_image(self, idx):
        if not (0 <= idx < len(self.image_files)):
            return

        path = self.image_files[idx]
        pix = QPixmap(path)

        if pix.isNull():
            QMessageBox.warning(self, "Load Failed", f"Failed to load image:\n{path}")
            return

        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self.pixmap_item)
        QTimer.singleShot(0, self._apply_initial_zoom)

        self.current_rot = 0
        self.view.setSceneRect(self.scene.itemsBoundingRect())
        # 默认放大策略：如果图片本身小于视口则使用实际大小（1.0），否则 Fit
        self._apply_initial_zoom()
        self.update_status()

    def _apply_initial_zoom(self):
        """Initial zoom: prefer actual size for small images, otherwise fit-to-window."""
        if not self.pixmap_item:
            return
        vb = self.view.viewport().rect()
        img_rect = self.pixmap_item.boundingRect().toRect()
        if img_rect.width() <= 0 or img_rect.height() <= 0:
            return
        sx = vb.width() / img_rect.width()
        sy = vb.height() / img_rect.height()
        scale = min(sx, sy)
        # if image is smaller than viewport, show at actual size (or slightly larger)
        if scale > 1.0:
            # image smaller -> display actual size (but if image is tiny, allow mild upscale upto scale)
            chosen = min(scale, 1.0 * max(1.0, scale))
        else:
            # image larger -> fit to window (keep small margin)
            chosen = max(0.01, scale * 0.98)

        # ensure finite positive
        if isfinite(chosen) and chosen > 0:
            # chosen = 1.0
            self.view.zoom_to(chosen)
        else:
            self.view.zoom_to(1.0)

    def resizeEvent(self, event):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def show_prev(self):
        if not self.image_files:
            return
        self.index = (self.index - 1) % len(self.image_files)
        self.load_image(self.index)

    def show_next(self):
        if not self.image_files:
            return
        self.index = (self.index + 1) % len(self.image_files)
        self.load_image(self.index)

    def fit_window(self):
        """适应窗口（安全版）"""
        if not self.pixmap_item:
            return
        img_rect = self.pixmap_item.boundingRect().toRect()
        if img_rect.width() == 0 or img_rect.height() == 0:
            return
        view_rect = self.view.viewport().rect()
        sx = view_rect.width() / img_rect.width()
        sy = view_rect.height() / img_rect.height()
        scale = min(sx, sy) * 0.98  # leave small margin
        if not isfinite(scale) or scale <= 0:
            scale = 1.0
        self.view.zoom_to(scale)
        self.update_status()

    def actual_size(self):
        self.view.zoom_to(1.0)
        self.update_status()

    def rotate(self, angle):
        if not self.pixmap_item:
            return
        self.current_rot = (self.current_rot + angle) % 360
        t = QTransform()
        t.rotate(self.current_rot)
        self.pixmap_item.setTransform(t)
        self.view.setSceneRect(self.scene.itemsBoundingRect())

    def update_status(self):
        if 0 <= self.index < len(self.image_files):
            self.label_name.setText(os.path.basename(self.image_files[self.index]))
        else:
            self.label_name.setText("")
        z = self.view.get_scale()
        if not isfinite(z) or z <= 0:
            z = 1.0
        self.label_zoom.setText(f"{int(z * 100)}%")

    # ---------------- Keyboard ----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last = event.pos()
        self.setFocus()  # 鼠标点中后抢焦点

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Right, Qt.Key_Down, Qt.Key_Space, Qt.Key_PageDown):
            self.show_next()
        elif key in (Qt.Key_Left, Qt.UpArrow, Qt.Key_PageUp):
            self.show_prev()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.view.zoom_in()
            self.update_status()
        elif key == Qt.Key_Minus:
            self.view.zoom_out()
            self.update_status()
        elif key == Qt.Key_F:
            self.fit_window()
        elif key == Qt.Key_A:
            self.actual_size()
        else:
            super().keyPressEvent(event)
