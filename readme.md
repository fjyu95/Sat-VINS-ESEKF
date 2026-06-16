# Sat-VINS-ESEKF

基于地理配准的VINS飞行定位技术方案

## 项目概述

本项目实现了一套基于高精度鲁棒地理配准 + ESEKF的完整VINS定位方案，主要用于无人机飞行定位场景。系统通过图像预处理、地理配准、LOFTR鲁棒配准和ESEKF融合等核心模块，实现高精度的视觉惯性导航定位。

## 核心功能

- **图像预处理**: 图像增强与去雾处理
- **地理配准**: 基于DINOV3 SAT-493M distilled特征的粗对齐
- **LOFTR鲁棒配准**: 自动过滤坏点的特征匹配
- **ESEKF融合**: 视觉定位与EKF融合，自监督过滤
- **可视化界面**: 基于PyQt5的可视化系统

## 技术路线

### 图像预处理
- **图像增强**: 经实测CLAHE方法最为稳定高效、视觉效果良好
  - 对比方法: GMM（需调参且计算慢）、直方图均衡化、AHE、DHE等
  - 深度学习方案参考: RetinexNet_PyTorch, rt-xnet, LoLi-IEA
- **图像去雾**: 
  - 暗通道法（不支持灰度图）
  - 深度学习: 实测PyTorch-Image-Dehazing（AOD-Net）速度快且效果良好
  - 参考项目: DCPDN, D4, FFA-Net, Dehamer, DehazeFormer

### 地理配准
- **旋转角度问题**: 基于DINOV3 SAT-493M distilled特征与地理北方向进行粗对齐
  - 当前支持90度连续旋转，可扩展为更小角度
  - 目前仅对首帧计算，大范围飞行和转弯时需再计算
- **尺度问题**: 自动计算GSD（根据相机焦距、高度、光轴角度），区分沿倾斜方法和垂直倾斜方向
- **搜索范围问题**: 底图自适应裁剪，影像金字塔多级检索（支持可变格式、不同数据源）

### LOFTR鲁棒配准
自动过滤坏点，基于以下指标：
- 凸性、面积、边长、边长比例
- 内角、内点数、内点比例等

### ESEKF融合
- **视觉定位**: PnP解算（cv2.solvePnPRansac + cv2.solvePnPRefineLM精修）
- **EKF融合**: 自监督过滤，与惯性预测结果相差过大时只用惯性，防止系统性崩溃

## 关键指标

- **图像配准**: 10px@0.6m
- **EKF融合**: RMSE<200m，图像增强后<150m

## 主要坐标系

- **WGS84经纬度**: EPSG:4326
- **Web墨卡托**: EPSG:3857
- **WGS84 ECEF**: EPSG:4978
- **高程系**: EPSG:5773
- **ENU坐标系**: 飞行首帧的ECEF坐标为原点
- **相机坐标系**
- **像素坐标系**

## 核心库

- **地理处理**: pymap3d, pyproj, rasterio
- **图像配准**: kornia, transformers
- **数据分析、绘图**: numpy, pandas, scipy, matplotlib, opencv, evo
- **可视化界面**: PyQt5

## 后续展望

### 鲁棒性
1. **旋转角度处理**: DINO特征具有一定的旋转不变性，可结合飞行先验、CLIP、LPIPS、ResNet、NCC、相位相关等方法综合判断
2. **弱纹理配准**: 结合光流、模板匹配、SuperPoint + LightGlue等方法
3. **多源卫星数据**: 利用多源卫星数据，避免单一数据源质量问题

### 实时性
1. 角度对齐、LOFTR配准显存占用和计算速度问题
2. 考虑云存储、云计算方案

### 准确性
1. 高分辨率（时间、空间）底图、DEM数据
2. 相机内参、多传感器外参的高精度标定
3. 光轴角度误差补偿
4. 首帧外的高精度GPS注入

## 安装

```bash
conda create -n geovio python=3.10
conda activate geovio
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
