"""轨迹生成域页面（对应 ui/samples/2_轨迹生成域GUI界面.png）。

三种模式：飞行轨迹生成（内置轨迹模板）/ 模拟自由飞 / 试飞数据回放。
生成结果绘制在「轨迹生成区域」，可储存为 CSV。功能自包含，与 VIO 导航解耦。
"""
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QComboBox, QPushButton, QGraphicsScene, QFileDialog, QMessageBox, QButtonGroup,
)
from PyQt5.QtGui import QPen, QColor, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QPointF

from ui.pages.common import ZoomableGraphicsView

SCENE_W, SCENE_H, MARGIN = 800.0, 460.0, 50.0


class TrajectoryPage(QMainWindow):
    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("轨迹生成域")
        self.resize(760, 680)

        self.traj_points = None   # 当前生成的轨迹点 (N, 2)
        self.replay_file = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---------------- 模式选择 ----------------
        mode_group = QGroupBox("模式选择")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_buttons = QButtonGroup(self)

        row1 = QHBoxLayout()
        self.radio_preset = QRadioButton("飞行轨迹生成")
        self.radio_preset.setChecked(True)
        self.combo_traj = QComboBox()
        self.combo_traj.addItems(["轨迹 1", "轨迹 2", "轨迹 3"])
        self.combo_traj.setMinimumWidth(160)
        row1.addWidget(self.radio_preset)
        row1.addWidget(self.combo_traj)
        row1.addStretch(1)
        mode_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.radio_free = QRadioButton("模拟自由飞")
        row2.addWidget(self.radio_free)
        row2.addStretch(1)
        mode_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.radio_replay = QRadioButton("试飞数据回放")
        self.btn_pick_file = QPushButton("选择文件路径...")
        self.btn_pick_file.clicked.connect(self.pick_replay_file)
        row3.addWidget(self.radio_replay)
        row3.addWidget(self.btn_pick_file)
        row3.addStretch(1)
        mode_layout.addLayout(row3)

        for radio in (self.radio_preset, self.radio_free, self.radio_replay):
            self.mode_buttons.addButton(radio)
        root.addWidget(mode_group)

        # ---------------- 功能 ----------------
        func_group = QGroupBox("功能")
        func_layout = QHBoxLayout(func_group)
        self.btn_generate = QPushButton("生成")
        self.btn_save = QPushButton("储存")
        self.btn_generate.clicked.connect(self.generate)
        self.btn_save.clicked.connect(self.save)
        func_layout.addWidget(self.btn_generate)
        func_layout.addWidget(self.btn_save)
        func_layout.addStretch(1)
        root.addWidget(func_group)

        # ---------------- 轨迹生成区域 ----------------
        draw_group = QGroupBox("轨迹生成区域")
        draw_layout = QVBoxLayout(draw_group)
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, SCENE_W, SCENE_H)
        self.view = ZoomableGraphicsView(self)
        self.view.setScene(self.scene)
        draw_layout.addWidget(self.view)
        root.addWidget(draw_group, stretch=1)

    # ------------------------------------------------------------------
    # 轨迹生成
    # ------------------------------------------------------------------
    def pick_replay_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择试飞数据文件", "results",
            "Trajectory Files (*.csv *.txt);;All Files (*)")
        if path:
            self.replay_file = path
            self.radio_replay.setChecked(True)
            self.btn_pick_file.setText(Path(path).name)
            self.btn_pick_file.setToolTip(path)

    def _generate_preset(self, idx):
        t = np.linspace(0.0, 1.0, 400)
        if idx == 0:      # 轨迹 1：S 形航线
            x = t * 4 * np.pi
            y = np.sin(x)
        elif idx == 1:    # 轨迹 2：环形航线
            a = t * 2 * np.pi
            x, y = np.cos(a), np.sin(a)
        else:             # 轨迹 3：螺旋航线
            a = t * 6 * np.pi
            r = 0.15 + t
            x, y = r * np.cos(a), r * np.sin(a)
        return np.column_stack([x, y])

    def _generate_free_fly(self):
        rng = np.random.default_rng()
        n = 400
        headings = np.cumsum(rng.normal(0.0, 0.05, n))   # 平滑随机航向
        steps = np.column_stack([np.cos(headings), np.sin(headings)])
        return np.cumsum(steps, axis=0)

    def _load_replay(self):
        if not self.replay_file:
            self.pick_replay_file()
            if not self.replay_file:
                return None
        data = np.loadtxt(self.replay_file)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("轨迹数据格式不正确（需要至少两列数据）")
        # 兼容 traj_esekf.csv（首列为时间戳）和纯 xy 两列格式
        return data[:, 1:3] if data.shape[1] >= 3 else data[:, :2]

    def generate(self):
        try:
            if self.radio_preset.isChecked():
                pts = self._generate_preset(self.combo_traj.currentIndex())
            elif self.radio_free.isChecked():
                pts = self._generate_free_fly()
            else:
                pts = self._load_replay()
                if pts is None:
                    return
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"轨迹生成失败！\n{e}", QMessageBox.Ok)
            return

        self.traj_points = np.asarray(pts, dtype=float)
        self._draw(self.traj_points)

    def save(self):
        if self.traj_points is None or len(self.traj_points) == 0:
            QMessageBox.information(self, "提示", "请先生成轨迹！", QMessageBox.Ok)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "储存轨迹", "results/generated_traj.csv", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, self.traj_points, fmt='%.6f', header='x y')
            QMessageBox.information(self, "储存成功", f"轨迹已保存至：\n{path}", QMessageBox.Ok)
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"轨迹储存失败！\n{e}", QMessageBox.Ok)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def _draw(self, pts):
        self.scene.clear()
        xy = pts[:, :2]
        if xy.shape[0] < 2:
            return

        # 归一化到场景坐标（y 轴翻转，保持地理直觉：y 向上）
        mn, mx = xy.min(axis=0), xy.max(axis=0)
        span = np.maximum(mx - mn, 1e-9)
        scale = min((SCENE_W - 2 * MARGIN) / span[0], (SCENE_H - 2 * MARGIN) / span[1])
        norm = (xy - mn) * scale + MARGIN
        norm[:, 1] = SCENE_H - norm[:, 1]

        path = QPainterPath(QPointF(norm[0, 0], norm[0, 1]))
        for x, y in norm[1:]:
            path.lineTo(x, y)
        self.scene.addPath(path, QPen(QColor(0, 120, 215), 2))

        # 起点（绿）/ 终点（红）标记
        r = 5
        self.scene.addEllipse(norm[0, 0] - r, norm[0, 1] - r, 2 * r, 2 * r,
                              pen=QPen(Qt.NoPen), brush=QBrush(QColor(0, 180, 0)))
        self.scene.addEllipse(norm[-1, 0] - r, norm[-1, 1] - r, 2 * r, 2 * r,
                              pen=QPen(Qt.NoPen), brush=QBrush(Qt.red))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
