# GeoVIO —— 无人相对导航全数字仿真验证环境

版本：v1.2
完成单位：西北工业大学

## 项目简介

GeoVIO 是一套面向无人机相对导航的全数字仿真验证环境。核心功能为视觉/惯性组合导航（Geo-VIO）：利用机载航拍影像与卫星基准图进行地理配准，结合 IMU 数据经误差状态扩展卡尔曼滤波（ESEKF）融合，实现无 GNSS 条件下的绝对定位与相对导航解算，并通过 PyQt5 图形界面完成全流程仿真与可视化。

## 功能逻辑

### 1. 视觉惯性导航主流程（`vio.py`）

逐帧处理航拍影像（`*.tif` 影像 + 同名 `*.txt` 元数据，GBK 编码），对每一帧执行：

1. **IMU 预测**：解析元数据中的 IMU 量测，`ESEKF_IMU/ekf_fusion.py` 驱动误差状态卡尔曼滤波器传播位姿，得到 IMU 预测位置（ENU 坐标）。
2. **首帧粗对准**：首帧通过 DINOv3（`visual_localization/ckpt/dinov3-vitl16-pretrain-sat493m`）提取航拍图与卫星底图的全局特征，用余弦相似度在 0/90/180/270 度中确定影像旋转角；也可在配置中直接指定 `rotation_angle`。
3. **基准图裁剪**：根据飞机位置、机下点高度与视场中心姿态角，推算视场覆盖的地面范围，在卫星影像金字塔中自动选取最接近地面分辨率的层级并裁剪对应区域。
4. **LoFTR 配准**：`visual_localization/geo_registration.py` 使用 LoFTR（outdoor 权重）对航拍图与裁剪底图做无特征点匹配，RANSAC 估计单应矩阵，并进行内点数、重投影误差及四边形几何有效性校验（凸性/面积/长宽比/内角）。
5. **PnP 定位解算**：`visual_localization/solve_uav.py` 将匹配点经 DEM 高程插值转为 ECEF 三维点，执行 PnP-RANSAC + LM 精化，解算相机绝对位置。
6. **EKF 融合**：PnP 结果与 IMU 预测偏差小于 `ekf_fusion_threshold` 时进行 ESEKF 更新；匹配失败或偏差过大时仅采用 IMU 预测结果，保证轨迹连续性。
7. **结果输出**：估计轨迹 `results/traj_esekf.csv`、真值轨迹 `results/traj_gt.csv`、匹配点 `image_matches_results/matched_points_*.npy`、配准统计 `results/register_stats.csv`、运行日志 `logs/`。

### 2. 相对导航仿真（`relative_navi/`）

主从机 TCP 通信仿真：`server.py` 读取 ESEKF 输出轨迹（ENU），经 `enu_datum.txt` 基准点转为 ECEF 坐标，以 0.5 s 周期向局域网发送；`client.py` 接收后按随机的距离/方位角/仰角解算僚机相对位置，并用 OpenCV 实时可视化双机位置。详细步骤见 `relative_navi/readme.md`。

### 3. 图像预处理（`preprocessing/`）

- `enhance/enhance.py`：CLAHE 自适应直方图均衡 / gamma 增强，提升弱光影像对比度。
- `dehaze/dehaze.py`：轻量 CNN 去雾网络（需权重 `preprocessing/dehaze/snapshots/dehazer.pth`）。

### 4. 结果统计分析（`stats_analysis/`）

- `navi_stats_plot.py`：对比 `traj_esekf.csv` 与 `traj_gt.csv`，绘制逐帧定位误差柱状+折线图，标注 Mean/RMSE/Std/SEM。
- `image_viewer.py`、`pandas_viewer.py`：图像与表格数据辅助查看工具。
- 亦可用 `evo_traj tum results/traj_esekf.csv --ref results/traj_gt.csv -p` 评估轨迹。

### 5. 仿真主界面（`main_window.py`）

PyQt5 界面按三个域组织，各按钮打开独立子窗口（`ui/pages/`）：

- **轨迹生成域**：轨迹生成/回放（`trajectory_page`）。
- **功能域**：机间链仿真、图像/视觉导航仿真、惯导/卫星导航仿真。
- **数据处理域**：数据预处理层、信息融合层（承载视觉/惯性融合导航 `NaviWindow`）、后置处理层。

菜单栏支持界面风格切换（Fusion/Windows/GTK+）、字体大小、深色主题、恢复默认配置。

## 目录结构

```
GeoVIO-v1.2/
├── main_window.py          # 仿真主界面入口
├── start.sh                # Linux 一键启动脚本（直接用 conda 环境 Python 启动主界面）
├── vio.py                  # 视觉惯性导航流水线（EKFWorker 线程）
├── configs/                # 配置：default.yaml / base.yaml / imu_parameters.yaml
├── ESEKF_IMU/              # 误差状态扩展卡尔曼滤波器（IMU 预测 + 量测更新）
├── visual_localization/    # 视觉定位：LoFTR 配准(geo_registration) + PnP(solve_uav)
│   └── ckpt/               # DINOv3 权重目录（需自行下载，不入库）
├── preprocessing/          # 图像增强(enhance)与去雾(dehaze)
├── relative_navi/          # 相对导航 TCP 仿真（server/client）
├── stats_analysis/         # 轨迹误差统计与数据查看工具
├── ui/                     # PyQt5 界面（pages/、样式表、界面样图 samples/）
├── utils/                  # 解析、地理处理、可视化、配置管理等公共模块
└── dataset/                # 数据集根目录（需自行准备，不入库）
```

## 环境安装

```bash
conda create -n geovio python=3.10
conda activate geovio
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

建议使用支持 CUDA 12.8 的 NVIDIA 显卡与驱动；无 GPU 时部分深度学习模块（DINOv3/LoFTR/去雾）将退化为 CPU 或不可用。

## 数据与权重准备

1. **数据集**：按 `configs/base.yaml` 中 `dataset_root_dir`（默认 `dataset/`）组织：
   - `aerial_images/`：航拍序列，每帧为 `*.tif` + 同名 `*.txt` 机载元数据（经纬度、海拔、IMU、视场角等，GBK 编码）。
   - `pyramid_images/`：卫星基准图瓦片金字塔，命名格式见 `tif_name_format`（默认 `tile_z{layer}_y{row}_x{col}.tif`）。
   - `DEM/`：数字高程模型（用于匹配点高程查询），路径见 `dem_path`。
   - `basemap/`：可视化底图（默认 `basemap/L16.tif`）。
2. **模型权重**：
   - DINOv3：`visual_localization/ckpt/dinov3-vitl16-pretrain-sat493m`（首帧旋转粗对准必需）。
   - 去雾：`preprocessing/dehaze/snapshots/dehazer.pth`（仅使用去雾功能时需要）。
   - LoFTR outdoor 权重由 kornia 自动下载。

## 操作说明

### 1. 配置参数

编辑 `configs/base.yaml`：

- **路径**：`dataset_root_dir`、`aerial_images_dir`、`pyramid_images_dir`、`dem_path`、`basemap_path`。
- **帧范围**：`start_frame_idx` / `end_frame_idx`（-1 表示到最后一帧）。
- **旋转角**：`rotation_angle`，可选 -1（自动估计）或 0/90/180/270。
- **相机**：`camera_intrinsics`、`camera_distortion`、`camera_axis_north_bias/east_bias`（光轴偏移）。
- **配准**：`scale_factor_crop_pyramid`（底图裁剪扩展倍数）、`register_ransacReprojThreshold`、`register_min_inliers`、`register_min_ratio`。
- **PnP**：`pnp_ransac_*`、`PnPRefineLM_*`。
- **EKF**：`ekf_sigma_p`（视觉定位精度，米）、`ekf_sigma_q`（弧度）、`ekf_fusion_threshold`（融合门限，米）。
- **调试**：`DEBUG: True` 仅处理前 30 帧。
- IMU 噪声参数在 `configs/imu_parameters.yaml`；界面菜单"设置 → 恢复默认配置文件"可重置为 `configs/default.yaml`。

### 2. 启动仿真主界面

```bash
python main_window.py
```

Linux 环境可使用一键启动脚本 `start.sh`（脚本内部直接调用 `~/miniconda3/envs/geovio/bin/python` 启动主界面，无需手动激活 conda 环境；如 conda 安装路径不同，请修改脚本中的 Python 路径）：

```bash
bash start.sh
```

依次打开：数据预处理层（增强/去雾）→ 信息融合层（视觉/惯性融合导航，`NaviWindow`）→ 后置处理层（轨迹与误差统计）。融合导航运行时可暂停/继续，底图上实时显示飞机位置与配准画面。

### 3. 相对导航仿真（双机通信）

```bash
# 主机（需先跑完导航流程生成 traj_esekf_out_f1.txt）
python relative_navi/server.py

# 从机（修改 client.py 中 SERVER_IP 为主机局域网 IP）
python relative_navi/client.py
```

按 `q` 退出可视化窗口。

### 4. 误差统计

```bash
cd stats_analysis
python navi_stats_plot.py   # 需将 traj_esekf.csv / traj_gt.csv 拷至该目录
```

## 常见问题

- **首帧旋转角估计失败/错误**：检查 DINOv3 权重是否存在，或直接在配置中指定 `rotation_angle`。
- **大量帧匹配失败**：检查金字塔分辨率层级覆盖范围、视场中心姿态角与 `camera_axis_*_bias` 设置，可适当增大 `scale_factor_crop_pyramid`。
- **输出目录**：`results/`、`logs/`、`image_matches_results/` 均为运行产物，已被 `.gitignore` 忽略。
