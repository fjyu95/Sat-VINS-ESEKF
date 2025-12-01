from dataclasses import dataclass, field, asdict
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML()
yaml.indent(sequence=4, offset=2)

yaml_text = """dataset_root_dir: "dataset"
aerial_images_dir: "aerial_images"
pyramid_images_dir: "pyramid_images"
dem_dir: "DEM"
basemap_dir: "basemap"

rotation_angle: -1 # optional: -1, 0, 90, 180, 270 degrees, clockwise
x_resolution: 0.125 # units: meters per pixel
y_resolution: 0.23
start_frame_idx: 0 # inclusive
end_frame_idx: -1 # inclusive

clahe_clip_limit: 2.0 # CLAHE clip limit for contrast enhancement

scale_factor_crop_pyramid: 1.5 # >1 to crop more
register_ransacReprojThreshold: 7.0 # loftr image registration ransac reprojection threshold
register_min_inliers: 20 # loftr image registration minimum inliers after ransac
register_min_ratio: 0.05 # loftr image registration minimum ratio after ransac

camera_intrinsics: [ 232707, 0, 2372, 0, 232707, 1897 ]
camera_distortion: [ 0.0, 0.0, 0.0, 0.0, 0.0 ]
pnp_ransac_reprojectionError: 8.0
pnp_ransac_confidence: 0.99
pnp_ransac_iterations: 500
PnPRefineLM_min_inliers: 10 # threshold to enable LM refinement
PnPRefineLM_maxIterations: 100

ekf_sigma_p: 5 # in meters
ekf_sigma_q: 0.015 # in rad
ekf_fusion_threshold: 1200.0 # threshold if use pnp result for ekf fusion, units: meters

DEBUG: False
"""


def restore_config(path="../config/default.yaml"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml_text, encoding="utf-8")
    print(f'restored config file to {Path(path).resolve()}')


@dataclass
class Config:
    # ----------------------
    # Dataset Paths
    # ----------------------
    dataset_root_dir: str = "dataset"
    aerial_images_dir: str = "aerial_images"
    pyramid_images_dir: str = "pyramid_images"
    dem_dir: str = "DEM"
    basemap_dir: str = "basemap"

    # ----------------------
    # Basic Parameters
    # ----------------------
    rotation_angle: int = -1  # -1, 0, 90, 180, 270 (clockwise)
    x_resolution: float = 0.125
    y_resolution: float = 0.23
    start_frame_idx: int = 0
    end_frame_idx: int = -1

    # ----------------------
    # Image Enhancement
    # ----------------------
    clahe_clip_limit: float = 2.0

    # ----------------------
    # Registration
    # ----------------------
    scale_factor_crop_pyramid: float = 1.5
    register_ransacReprojThreshold: float = 7.0
    register_min_inliers: int = 20
    register_min_ratio: float = 0.05

    # ----------------------
    # Camera
    # ----------------------
    camera_intrinsics: list = field(default_factory=lambda: [232707, 0, 2372, 0, 232707, 1897])
    camera_distortion: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # ----------------------
    # PnP
    # ----------------------
    pnp_ransac_reprojectionError: float = 8.0
    pnp_ransac_confidence: float = 0.99
    pnp_ransac_iterations: int = 500
    PnPRefineLM_min_inliers: int = 10
    PnPRefineLM_maxIterations: int = 100

    # ----------------------
    # EKF
    # ----------------------
    ekf_sigma_p: float = 5.0
    ekf_sigma_q: float = 0.015
    ekf_fusion_threshold: float = 1200.0

    # ----------------------
    # Debug
    # ----------------------
    DEBUG: bool = False

    # ---------------------------------------------------------
    # Save to YAML
    # ---------------------------------------------------------
    def save(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f)

    # ---------------------------------------------------------
    # Load from YAML (overrides defaults)
    # ---------------------------------------------------------
    @classmethod
    def load(cls, filepath: str):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f)

        # 将 YAML 内容填入 dataclass
        return cls(**data)


def test():
    restore_config('default.yaml')

    cfg = Config.load("../config/base.yaml")
    print(cfg.camera_intrinsics)
    print(cfg.rotation_angle)

    cfg = Config()
    cfg.DEBUG = False
    cfg.x_resolution = 0.20
    cfg.camera_intrinsics[0] = 230000
    cfg.save("base.yaml")


if __name__ == "__main__":
    pass
    # test()
