"""数据处理域三个层页面（对应 ui/samples/6_数据处理层GUI界面.png）。

三个独立页面分别对应数据处理域三层，按钮与系统真实功能一一对应：
- DataPreprocessPage（数据预处理层）：读入数据 -> 数据预处理(去雾+增强) -> 发送至融合层
- InfoFusionPage（信息融合层）：融合方案配置 -> 信息融合(打开导航窗口) -> 发送至后置处理层
- PostProcessPage（后置处理层）：误差统计 / 误差曲线 / 结果比较
"""
from pathlib import Path

import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QLabel, QMenu,
)

from utils.parser import load_config
from preprocessing.dehaze.dehaze import dehaze_image
from preprocessing.enhance.enhance import clahe_enhance
from ui.pages.context import IMAGE_EXTS
from ui.pages.workers import run_preprocess
from ui.pages.common import (
    show_match_results, show_registration_stats, show_est_vs_gt_traj, compute_ekf_error_stats,
)


class _DataLayerPage(QMainWindow):
    """数据处理域页面基类：统一按钮样式与状态栏。"""

    def __init__(self, ctx, title, height=300, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(title)
        self.resize(480, height)

        self._worker = None
        self._viewers = []

        central = QWidget()
        self.setCentralWidget(central)
        self._root = QVBoxLayout(central)
        self._root.setContentsMargins(12, 12, 12, 12)
        self._root.setSpacing(10)

    def _finish_layout(self):
        """在子类添加完功能按钮后调用：追加状态栏与弹性空间。"""
        self.status_label = QLabel("状态：待机")
        self._root.addWidget(self.status_label)
        self._root.addStretch(1)

    @staticmethod
    def _make_button(text, slot):
        btn = QPushButton(text)
        btn.setMinimumHeight(32)
        btn.clicked.connect(slot)
        return btn

    def _set_status(self, text):
        self.status_label.setText(f"状态：{text}")
        print(f">>> {text}")


class DataPreprocessPage(_DataLayerPage):
    """数据预处理层页面：读入数据 -> 数据预处理 -> 发送至融合层。"""

    def __init__(self, ctx, open_fusion=None, parent=None):
        super().__init__(ctx, "数据预处理层", 300, parent)
        self.open_fusion = open_fusion   # 由主界面注入：打开信息融合层页面

        pre_group = QGroupBox("数据预处理层")
        pre_layout = QVBoxLayout(pre_group)
        pre_layout.addWidget(self._make_button("读入数据", self.load_data))

        # 「数据预处理」为下拉菜单按钮，包含图像去雾 / 图像增强两个下级选项
        btn_preprocess = QPushButton("数据预处理")
        btn_preprocess.setMinimumHeight(32)
        menu = QMenu(self)
        menu.addAction("图像去雾", self.btn_dehaze)
        menu.addAction("图像增强", self.btn_enhance)
        btn_preprocess.setMenu(menu)
        pre_layout.addWidget(btn_preprocess)

        pre_layout.addWidget(self._make_button("发送至融合层", self.send_to_fusion))
        self._root.addWidget(pre_group)
        self._finish_layout()

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

    # 只能对原图去雾，不能先增强再去雾
    def btn_dehaze(self):
        images_dir = Path(self.ctx.config['aerial_images_dir'])
        if self.ctx.config['DEBUG']:
            images_dir = Path('preprocessing/dehaze/test_data')  # for debug
        output_dir = images_dir.parent.resolve() / 'dehazed'

        sorted_files = self._list_images(images_dir)
        if not sorted_files:
            QMessageBox.warning(self, "提示", f"未在目录中找到图像：\n{images_dir}", QMessageBox.Ok)
            return

        try:
            img = cv2.imread(sorted_files[0], -1)
            if img.ndim == 2:
                QMessageBox.information(self, "操作无效", f"暂不支持灰度图像去雾！\n{images_dir}", QMessageBox.Ok)
                return
        except Exception as e:
            print(f'Exception {images_dir}, {e}')
            return

        self._set_status("图像去雾中……")
        self._worker = run_preprocess(self, sorted_files, output_dir.as_posix(),
                                      dehaze_image, 'dehazed', on_finished=self.handle_results)

    def btn_enhance(self):
        if self.ctx.dehazed_dir:
            images_dir = Path(self.ctx.dehazed_dir)
        else:
            images_dir = Path(self.ctx.config['aerial_images_dir'])
        output_dir = images_dir.parent.resolve() / 'enhanced'

        # 若已有去雾结果，只增强去雾过的图像，过滤 combined
        prefix = 'dehazed_' if self.ctx.dehazed_dir else None
        sorted_files = self._list_images(images_dir, prefix=prefix)
        if not sorted_files:
            QMessageBox.warning(self, "提示", f"未在目录中找到图像：\n{images_dir}", QMessageBox.Ok)
            return

        kwargs = {'clip_limit': self.ctx.config['clahe_clip_limit']}
        self._set_status("图像增强中……")
        self._worker = run_preprocess(self, sorted_files, output_dir.as_posix(),
                                      clahe_enhance, 'enhanced',
                                      on_finished=self.handle_results, **kwargs)

    def handle_results(self, func_name, output_dir):
        if func_name == 'dehazed':
            self.ctx.dehazed_dir = output_dir
            self._set_status(f"图像去雾完成：{output_dir}")
            QMessageBox.information(self, "处理成功", f"图像去雾结果保存在 {output_dir}", QMessageBox.Ok)
        elif func_name == 'enhanced':
            self.ctx.enhanced_dir = output_dir
            self._set_status(f"图像增强完成：{output_dir}")
            QMessageBox.information(self, "处理成功", f"图像增强结果保存在 {output_dir}", QMessageBox.Ok)
        else:
            print(f'Unknown function name: {func_name}')

        print(f'>>> newest images switch to {output_dir}')

    def send_to_fusion(self):
        """发送至融合层：将最新预处理结果登记为导航输入，并打开信息融合层页面。"""
        files = self.ctx.set_final_images()
        if files:
            self._set_status(f"已发送至融合层（{len(files)} 帧预处理图像）")
            QMessageBox.information(self, "发送至融合层",
                                    f"已选择 {len(files)} 帧预处理图像作为融合输入。", QMessageBox.Ok)
        else:
            self._set_status("已发送至融合层（使用原始航拍图像）")
            QMessageBox.information(self, "发送至融合层",
                                    "未检测到预处理结果，将使用原始航拍图像进行融合。", QMessageBox.Ok)
        if self.open_fusion is not None:
            self.open_fusion()


class InfoFusionPage(_DataLayerPage):
    """信息融合层页面：融合方案配置 -> 信息融合 -> 发送至后置处理层。"""

    def __init__(self, ctx, open_navi=None, open_post=None, parent=None):
        super().__init__(ctx, "信息融合层", 300, parent)
        self.open_navi = open_navi   # 由主界面注入：打开 NaviWindow
        self.open_post = open_post   # 由主界面注入：打开后置处理层页面

        fusion_group = QGroupBox("信息融合层")
        fusion_layout = QVBoxLayout(fusion_group)
        for text, slot in (("融合方案配置", self.fusion_config),
                           ("信息融合", self.start_fusion),
                           ("发送至后置处理层", self.send_to_post)):
            fusion_layout.addWidget(self._make_button(text, slot))
        self._root.addWidget(fusion_group)
        self._finish_layout()

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
            if self.open_post is not None:
                self.open_post()
        else:
            QMessageBox.warning(self, "发送至后置处理层",
                                "尚未检测到融合结果文件（results/traj_esekf.csv），\n请先运行「信息融合」。",
                                QMessageBox.Ok)


class PostProcessPage(_DataLayerPage):
    """后置处理层页面：误差统计 / 误差曲线 / 结果比较及故障分析。"""

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, "后置处理层", 300, parent)

        post_group = QGroupBox("后置处理层")
        post_layout = QVBoxLayout(post_group)
        for text, slot in (("仿真计算误差", self.calc_error),
                           ("绘制误差曲线", self.plot_error),
                           ("结果比较及故障分析", self.compare_results)):
            post_layout.addWidget(self._make_button(text, slot))
        self._root.addWidget(post_group)
        self._finish_layout()

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
