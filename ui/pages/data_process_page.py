"""数据处理层页面（对应 ui/samples/6_数据处理层GUI界面.png）。

三层按钮与系统真实功能一一对应：
- 数据预处理层：读入数据(加载配置) -> 数据预处理(去雾+增强) -> 发送至融合层
- 信息融合层：融合方案配置 -> 信息融合(打开导航窗口) -> 发送至后置处理层
- 后置处理层：误差统计 / 误差曲线 / 结果比较
"""
from pathlib import Path

import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QLabel,
)

from utils.parser import load_config
from preprocessing.dehaze.dehaze import dehaze_image
from preprocessing.enhance.enhance import clahe_enhance
from ui.pages.context import IMAGE_EXTS
from ui.pages.workers import run_preprocess
from ui.pages.common import (
    show_match_results, show_registration_stats, show_est_vs_gt_traj, compute_ekf_error_stats,
)


class DataProcessPage(QMainWindow):
    def __init__(self, ctx, open_navi=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.open_navi = open_navi   # 由主界面注入：打开 NaviWindow
        self.setWindowTitle("数据处理层")
        self.resize(480, 620)

        self._worker = None
        self._viewers = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---------------- 数据预处理层 ----------------
        pre_group = QGroupBox("数据预处理层")
        pre_layout = QVBoxLayout(pre_group)
        self.btn_load_data = self._make_button("读入数据", self.load_data)
        self.btn_preprocess = self._make_button("数据预处理", self.preprocess)
        self.btn_send_fusion = self._make_button("发送至融合层", self.send_to_fusion)
        for b in (self.btn_load_data, self.btn_preprocess, self.btn_send_fusion):
            pre_layout.addWidget(b)
        root.addWidget(pre_group)

        # ---------------- 信息融合层 ----------------
        fusion_group = QGroupBox("信息融合层")
        fusion_layout = QVBoxLayout(fusion_group)
        self.btn_fusion_cfg = self._make_button("融合方案配置", self.fusion_config)
        self.btn_fusion = self._make_button("信息融合", self.start_fusion)
        self.btn_send_post = self._make_button("发送至后置处理层", self.send_to_post)
        for b in (self.btn_fusion_cfg, self.btn_fusion, self.btn_send_post):
            fusion_layout.addWidget(b)
        root.addWidget(fusion_group)

        # ---------------- 后置处理层 ----------------
        post_group = QGroupBox("后置处理层")
        post_layout = QVBoxLayout(post_group)
        self.btn_err_calc = self._make_button("仿真计算误差", self.calc_error)
        self.btn_err_plot = self._make_button("绘制误差曲线", self.plot_error)
        self.btn_compare = self._make_button("结果比较及故障分析", self.compare_results)
        for b in (self.btn_err_calc, self.btn_err_plot, self.btn_compare):
            post_layout.addWidget(b)
        root.addWidget(post_group)

        # 状态提示
        self.status_label = QLabel("状态：待机")
        root.addWidget(self.status_label)
        root.addStretch(1)

    @staticmethod
    def _make_button(text, slot):
        btn = QPushButton(text)
        btn.setMinimumHeight(32)
        btn.clicked.connect(slot)
        return btn

    def _set_status(self, text):
        self.status_label.setText(f"状态：{text}")
        print(f">>> {text}")

    # ==================================================================
    # 数据预处理层
    # ==================================================================
    def load_data(self):
        """读入数据：选择并加载仿真配置文件。"""
        if not self.ctx.config:
            QMessageBox.warning(self, "提示", "尚未加载配置文件，请通过「融合方案配置」选择配置文件。",
                                QMessageBox.Ok)
            return
        images_dir = Path(self.ctx.config['aerial_images_dir'])
        n_images = len([p for p in images_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0
        QMessageBox.information(
            self, "读入数据",
            f"航拍图像目录：{images_dir}\n共 {n_images} 帧图像\n"
            f"底图：{self.ctx.config['basemap_path']}",
            QMessageBox.Ok)
        self._set_status(f"数据已读入（{n_images} 帧）")

    @staticmethod
    def _list_images(images_dir, prefix=None):
        files = [str(p) for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if prefix:
            files = [p for p in files if Path(p).name.startswith(prefix)]
        return sorted(files)

    def preprocess(self):
        """数据预处理：先去雾、再增强（链式执行，结果写入共享上下文）。"""
        if not self.ctx.config:
            QMessageBox.warning(self, "提示", "请先通过「融合方案配置」加载配置文件。", QMessageBox.Ok)
            return

        images_dir = Path(self.ctx.config['aerial_images_dir'])
        if self.ctx.config['DEBUG']:
            images_dir = Path('preprocessing/dehaze/test_data')  # for debug
        if not images_dir.exists():
            QMessageBox.warning(self, "提示", f"航拍图像目录不存在：\n{images_dir}", QMessageBox.Ok)
            return

        sorted_files = self._list_images(images_dir)
        if not sorted_files:
            QMessageBox.warning(self, "提示", f"未在目录中找到图像：\n{images_dir}", QMessageBox.Ok)
            return

        try:
            img = cv2.imread(sorted_files[0], -1)
            if img is not None and img.ndim == 2:
                QMessageBox.information(self, "操作无效", f"暂不支持灰度图像去雾！\n{images_dir}", QMessageBox.Ok)
                return
        except Exception as e:
            print(f'Exception {images_dir}, {e}')
            return

        output_dir = images_dir.parent.resolve() / 'dehazed'
        self._set_status("图像去雾中……")
        self._worker = run_preprocess(self, sorted_files, output_dir.as_posix(),
                                      dehaze_image, 'dehazed', on_finished=self._on_dehazed)

    def _on_dehazed(self, func_name, output_dir):
        self.ctx.dehazed_dir = output_dir
        print(f'>>> dehazed images saved to {output_dir}')

        # 链式执行图像增强：仅处理 dehazed_* 结果
        images_dir = Path(output_dir)
        enhance_dir = images_dir.parent.resolve() / 'enhanced'
        files = self._list_images(images_dir, prefix='dehazed_')
        if not files:
            self._set_status("去雾完成（无结果用于增强）")
            return

        kwargs = {'clip_limit': self.ctx.config['clahe_clip_limit']}
        self._set_status("图像增强中……")
        self._worker = run_preprocess(self, files, enhance_dir.as_posix(),
                                      clahe_enhance, 'enhanced',
                                      on_finished=self._on_enhanced, **kwargs)

    def _on_enhanced(self, func_name, output_dir):
        self.ctx.enhanced_dir = output_dir
        self._set_status(f"预处理完成，结果：{output_dir}")
        QMessageBox.information(self, "处理成功", f"图像预处理完成，结果保存在：\n{output_dir}", QMessageBox.Ok)

    def send_to_fusion(self):
        """发送至融合层：将最新预处理结果登记为导航输入。"""
        files = self.ctx.set_final_images()
        if files:
            self._set_status(f"已发送至融合层（{len(files)} 帧预处理图像）")
            QMessageBox.information(self, "发送至融合层",
                                    f"已选择 {len(files)} 帧预处理图像作为融合输入。", QMessageBox.Ok)
        else:
            self._set_status("已发送至融合层（使用原始航拍图像）")
            QMessageBox.information(self, "发送至融合层",
                                    "未检测到预处理结果，将使用原始航拍图像进行融合。", QMessageBox.Ok)

    # ==================================================================
    # 信息融合层
    # ==================================================================
    def fusion_config(self):
        """融合方案配置：选择并加载 yaml 配置文件。"""
        default_dir = Path('configs').resolve().as_posix()
        path, _ = QFileDialog.getOpenFileName(self, "选择融合配置文件", default_dir, "Config Files (*.yaml)")
        if not path:
            return
        try:
            config = load_config(path)
            if not config:
                QMessageBox.critical(self, "错误", f"加载配置文件失败！请检查\n{path}", QMessageBox.Ok)
                return
            self.ctx.config = config
            summary = (f"配置文件：{path}\n"
                       f"图像分辨率：{config['x_resolution']} / {config['y_resolution']} m/px\n"
                       f"帧范围：{config['start_frame_idx']} ~ {config['end_frame_idx']}\n"
                       f"EKF融合阈值：{config['ekf_fusion_threshold']} m")
            QMessageBox.information(self, "融合方案配置", f"配置已加载：\n{summary}", QMessageBox.Ok)
            self._set_status("融合方案已配置")
        except Exception as e:
            print("Error loading config file: ", path, e)
            QMessageBox.critical(self, "错误", f"加载配置文件失败！请检查\n{path}", QMessageBox.Ok)

    def start_fusion(self):
        """信息融合：打开视觉/惯性融合导航窗口。"""
        if not self.ctx.config:
            QMessageBox.warning(self, "提示", "请先通过「融合方案配置」加载配置文件。", QMessageBox.Ok)
            return
        if self.open_navi is not None:
            self.open_navi()
            self._set_status("信息融合中（导航仿真窗口已打开）")

    def send_to_post(self):
        """发送至后置处理层：检查融合输出轨迹文件是否就绪。"""
        est = Path('results/traj_esekf.csv')
        gt = Path('results/traj_gt.csv')
        if est.exists() and gt.exists():
            self._set_status("融合结果已发送至后置处理层")
            QMessageBox.information(self, "发送至后置处理层",
                                    f"融合轨迹已就绪：\n{est.resolve()}\n{gt.resolve()}", QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "发送至后置处理层",
                                "尚未检测到融合结果文件（results/traj_esekf.csv），\n请先运行「信息融合」。",
                                QMessageBox.Ok)

    # ==================================================================
    # 后置处理层
    # ==================================================================
    def calc_error(self):
        """仿真计算误差：统计 ESEKF 定位误差并保存分布图。"""
        path = compute_ekf_error_stats(self)
        if path:
            self._set_status(f"误差统计完成：{path}")

    def plot_error(self):
        """绘制误差曲线：估计轨迹 vs 真值轨迹。"""
        show_est_vs_gt_traj(self)
        self._set_status("误差曲线已绘制")

    def compare_results(self):
        """结果比较及故障分析：配准图像浏览 + 配准统计表。"""
        v1 = show_match_results(self)
        v2 = show_registration_stats(self)
        self._viewers.extend([v for v in (v1, v2) if v is not None])
        self._set_status("结果比较窗口已打开")
