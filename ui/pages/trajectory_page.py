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
    def __init__(self, ctx=None, open_navi=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.open_navi = open_navi   # 由主界面注入：打开导航仿真窗口
        self.setWindowTitle("轨迹生成域")
        self.resize(760, 680)

        self.traj_points = None   # 当前生成的轨迹点 (N, 2)
        self.traj_name = None     # 当前轨迹名称（用于导航界面叠加显示）
        self.gen_count = 0
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
        self.combo_traj.addItems(["轨迹 1（测绘航线）", "轨迹 2（盘旋航线）", "轨迹 3（航点航线）"])
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
        self.btn_to_navi = QPushButton("添加至导航界面")
        self.btn_generate.clicked.connect(self.generate)
        self.btn_save.clicked.connect(self.save)
        self.btn_to_navi.clicked.connect(self.add_to_navi)
        func_layout.addWidget(self.btn_generate)
        func_layout.addWidget(self.btn_save)
        func_layout.addWidget(self.btn_to_navi)
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
        if idx == 0:
            return self._gen_survey()
        elif idx == 1:
            return self._gen_orbit()
        else:
            return self._gen_waypoints()

    @staticmethod
    def _gen_survey(n_pass=6, length=6.0, spacing=0.8):
        """测绘往返航线：平行航段 + 端部半圆掉头（覆盖式航拍典型航线）。"""
        n_leg, n_arc = 60, 24
        pts = []
        for i in range(n_pass):
            y = i * spacing
            xs = np.linspace(0.0, length, n_leg) if i % 2 == 0 \
                else np.linspace(length, 0.0, n_leg)
            pts.append(np.column_stack([xs, np.full(n_leg, y)]))
            if i < n_pass - 1:
                cx = length if i % 2 == 0 else 0.0
                cy = y + spacing / 2.0
                a = np.linspace(-np.pi / 2, np.pi / 2, n_arc) if i % 2 == 0 \
                    else np.linspace(-np.pi / 2, -3 * np.pi / 2, n_arc)
                pts.append(np.column_stack([cx + spacing / 2 * np.cos(a),
                                            cy + spacing / 2 * np.sin(a)]))
        return np.vstack(pts)

    @staticmethod
    def _gen_orbit(r=1.2, n_orb=4):
        """盘旋航线：直线进入 + 多圈定圆盘旋 + 直线退出（巡逻/待命典型航线）。"""
        cx, cy = 3.0, 0.0
        entry = np.column_stack([np.linspace(0.0, cx, 40), np.full(40, cy - r)])
        a = np.linspace(-np.pi / 2, -np.pi / 2 + n_orb * 2 * np.pi, n_orb * 60)
        orbit = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
        ext = np.column_stack([np.linspace(cx, cx + 3.0, 40), np.full(40, cy - r)])
        return np.vstack([entry, orbit, ext])

    def _gen_waypoints(self):
        """航点航线：直线航段 + 限转弯半径圆弧过渡（多点任务航线）。"""
        wps = np.array([[0.0, 0.0], [3.0, 0.4], [4.6, 1.8], [3.4, 3.4],
                        [1.2, 3.0], [0.4, 1.6], [1.8, 0.6]])
        return self._smooth_route(wps, turn_r=0.45)

    @staticmethod
    def _smooth_route(wps, turn_r, n_leg=40, n_arc=18):
        """航点折线 → 带圆弧转弯的平滑航线（简化 Dubins 路径）。"""
        wps = np.asarray(wps, dtype=float)
        segs = []
        start = wps[0]
        for i in range(1, len(wps) - 1):
            prev, cur, nxt = wps[i - 1], wps[i], wps[i + 1]
            u = cur - prev
            nu = np.linalg.norm(u)
            v = nxt - cur
            nv = np.linalg.norm(v)
            if nu < 1e-9 or nv < 1e-9:
                continue
            u, v = u / nu, v / nv
            delta = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))  # 转弯角
            if delta < 1e-3:
                continue
            delta = min(delta, np.pi * 0.99)  # 避免 180° 掉头时正切发散
            d = min(turn_r / np.tan(delta / 2.0), 0.45 * nu, 0.45 * nv)
            t1 = cur - u * d
            t2 = cur + v * d
            segs.append(np.linspace(start, t1, n_leg))
            # 转弯圆弧：圆心在内角平分线上，与两航段相切
            r = d * np.tan(delta / 2.0)
            m = v - u
            m /= np.linalg.norm(m)
            o = cur + m * (d / np.cos(delta / 2.0))
            a1 = np.arctan2(t1[1] - o[1], t1[0] - o[0])
            a2 = np.arctan2(t2[1] - o[1], t2[0] - o[0])
            da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
            a = np.linspace(a1, a1 + da, n_arc)
            segs.append(np.column_stack([o[0] + r * np.cos(a),
                                         o[1] + r * np.sin(a)]))
            start = t2
        segs.append(np.linspace(start, wps[-1], n_leg))
        return np.vstack(segs)

    def _generate_free_fly(self):
        """模拟自由飞：随机航点 + 恒定速度/限转率追踪，生成平滑飞行轨迹。"""
        rng = np.random.default_rng()
        wps = rng.uniform(-4.0, 4.0, (7, 2))
        pos = np.array([0.0, 0.0])
        heading = rng.uniform(0.0, 2 * np.pi)
        v, omega_max, dt, tol = 0.12, 1.5, 0.1, 0.15  # 速度/最大转弯率/步长/到达阈值
        pts = []
        for wp in wps:
            for _ in range(2000):
                if np.linalg.norm(wp - pos) <= tol:
                    break
                desired = np.arctan2(wp[1] - pos[1], wp[0] - pos[0])
                err = (desired - heading + np.pi) % (2 * np.pi) - np.pi
                heading += np.clip(err, -omega_max * dt, omega_max * dt)
                pos = pos + v * dt * np.array([np.cos(heading), np.sin(heading)])
                pts.append(pos.copy())
        return np.asarray(pts)

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
                idx = self.combo_traj.currentIndex()
                pts = self._generate_preset(idx)
                base = ['测绘航线', '盘旋航线', '航点航线'][idx]
            elif self.radio_free.isChecked():
                pts = self._generate_free_fly()
                base = '自由飞'
            else:
                pts = self._load_replay()
                if pts is None:
                    return
                base = f'回放-{Path(self.replay_file).stem}'
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "错误", f"轨迹生成失败！\n{e}", QMessageBox.Ok)
            return

        self.gen_count += 1
        self.traj_name = f'{base}#{self.gen_count}'
        self.traj_points = np.asarray(pts, dtype=float)
        self._draw(self.traj_points)

    def add_to_navi(self):
        """生成后直接添加：登记到共享上下文并叠加显示到导航底图。"""
        if self.traj_points is None or len(self.traj_points) == 0:
            QMessageBox.information(self, "提示", "请先生成轨迹！", QMessageBox.Ok)
            return
        if self.ctx is None:
            QMessageBox.warning(self, "提示", "无共享上下文，无法传递轨迹。", QMessageBox.Ok)
            return
        self.ctx.add_trajectory(self.traj_name, self.traj_points)
        if self.open_navi is None:
            QMessageBox.information(self, "已登记",
                                    f"轨迹「{self.traj_name}」已登记，可在导航界面选择性添加。",
                                    QMessageBox.Ok)
            return
        win = self.open_navi()
        win.refresh_trajectories()

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
