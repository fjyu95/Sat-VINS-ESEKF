"""应用共享上下文。

配置文件、数据预处理结果目录等状态在主界面与各功能子页面之间共享，
保证「数据预处理 -> 信息融合 -> 后置处理」流程中各页面看到一致的数据。
"""
from pathlib import Path

from utils.parser import load_config

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']


class AppContext:
    """跨页面共享的仿真上下文。"""

    def __init__(self):
        self.config = load_config()      # 当前生效配置（dict）
        self.dehazed_dir = None          # 最近一次图像去雾输出目录
        self.enhanced_dir = None         # 最近一次图像增强输出目录

    @property
    def basemap_path(self):
        return self.config['basemap_path'] if self.config else None

    def reload_config(self, path="configs/base.yaml"):
        """重新加载配置文件，成功时替换当前配置。"""
        config = load_config(path)
        if config:
            self.config = config
        return config

    def set_final_images(self):
        """挑选最新的预处理结果作为视觉导航输入。

        优先级：增强结果 > 去雾结果 > 原始航拍图像（final_files=None）。
        返回图像文件列表（Path 对象），无预处理结果时返回 None。
        """
        if self.enhanced_dir:
            files = [p for p in Path(self.enhanced_dir).iterdir()
                     if p.suffix.lower() in IMAGE_EXTS and p.name.startswith('enhanced_')]
        elif self.dehazed_dir:
            files = [p for p in Path(self.dehazed_dir).iterdir()
                     if p.suffix.lower() in IMAGE_EXTS and p.name.startswith('dehazed_')]
        else:
            if self.config:
                self.config['final_files'] = None
            return None

        self.config['final_files'] = sorted(files)
        return self.config['final_files']
