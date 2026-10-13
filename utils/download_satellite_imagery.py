import asyncio
import glob
import math
import os
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import rasterio
import pymap3d as pm
import pyproj
from pyproj import Transformer, CRS
from rasterio.transform import rowcol, from_bounds
from rasterio.merge import merge
import rasterio.windows as windows
from skimage.draw import ellipsoid_stats
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
import requests
import mercantile

from utils.core_utils import timer

proxy = "socks5://127.0.0.1:1083"
http_proxy = "http://127.0.0.1:8889"

proxies = {
    "http": "socks5://127.0.0.1:1083",
    "https": "socks5://127.0.0.1:1083",  # https 也走 SOCKS
}  # 超时可以尝试更换节点

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}


def get_url_template(source_type='google'):
    url_template = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    if source_type == "arcgis":
        # ArcGIS 使用标准的 XYZ
        url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    elif source_type == "bing":
        # Bing 需要计算 QuadKey
        url_template = "http://ecn.t1.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
    elif source_type == 'mapbox':
        url_template = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=YOUR_ACCESS_TOKEN'
    elif source_type == 'tianditu':
        # 虽然是官方地图，但在公共 Web 服务中，它的瓦片切割原点和坐标系（CGCS2000）在某些缩放级别下与标准的 Google Web Mercator 存在像素级错位，且现在需要强制申请 Key
        url_template = (
            "http://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default"
            "&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={tk}")

    return url_template


# 需要设置http/https或socks代理
# 最好生成标准TIFF格式,否则处理起来可能比较麻烦,而且通用性不好
def download_satellite_imagery(source_type='google'):
    """
    下载 1 张谷歌卫星瓦片（RGB，亚米级）
    自动生成 WorldFile (.tfw)  →  GIS 软件直接打开
    """
    import mercantile, requests, io
    from PIL import Image

    # os.environ["http_proxy"] = "http://127.0.0.1:8889"
    # os.environ["https_proxy"] = "http://127.0.0.1:8889"

    # ========= 用户参数 =========
    lat, lon = 39.909604, 116.397228  # 天安门中心
    zoom = 18  # ≈ 0.6 m/像素
    out_dir = Path("tile")
    # ============================

    out_dir.mkdir(exist_ok=True)
    tile = mercantile.tile(lon, lat, zoom)  # 瓦片编号
    x, y, z = tile.x, tile.y, tile.z
    tif_path = out_dir / f"sat_z{z}_x{x}_y{y}.tif"
    tif_path = 'merged.tif'
    # tif_path = '11_823_458_01-01.tif'
    # tif_path = 'ALPSMLC30_N018E109_DSM.tif'

    if os.path.exists(tif_path):
        with rasterio.open(tif_path) as src:
            res_x = src.transform.a  # 像素宽度
            res_y = -src.transform.e  # 像素高度（通常为负数）
            size_ = src.shape[0]
            x0, y0 = src.xy(0, 0, offset='ul')
            transform = src.transform
            width = src.width
            height = src.height
            crs = src.crs
            gcps = src.gcps
            rpcs = src.rpcs
            print(crs)

    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"  # 谷歌卫星 RGB 通道
    if source_type == "google":
        size = 512  # 256/512/640/1024
        url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&size={size}"  # 使用默认的256也完全没有问题,并且下载成功率高
        url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&scale=2"  # 不能随意调整，只能在 256 和 512 之间选择

    elif source_type == "arcgis":
        # ArcGIS 使用标准的 XYZ
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    elif source_type == "bing":
        # Bing 需要计算 QuadKey
        q = mercantile.quadkey(tile)
        subdomain = 1  # subdomains: t0, t1, t2, t3 可以随机选一个
        # 'a' 代表 Aerial (卫星图), 'h' 代表 Hybrid (混合), 'r' 代表 Road (路网)
        url_template = f"https://ecn.t{subdomain}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
        url = f"http://ecn.t1.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"

    print("Downloading", url)

    # resp = requests.get(url, timeout=30)
    resp = requests.get(url, proxies=proxies, timeout=30)
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content))
    print("Received size:", img.size)

    img.save(tif_path, format="TIFF")
    print("Saved RGB :", tif_path)
    return

    # 计算左上角地理坐标（WGS84）
    west, south, east, north = mercantile.bounds(x, y, z)
    pixel_size = 156543.03 * math.cos(math.radians(north)) / (2 ** zoom)  # 米/像素

    # 写出 WorldFile (.tfw)  →  QGIS/ArcGIS 可识别
    tfw_path = tif_path.with_suffix(".tfw")
    with tfw_path.open("w") as f:
        # 行列方向 & 左上角中心
        f.write(f"{pixel_size}\n")  # A
        f.write("0.0\n")  # B
        f.write("0.0\n")  # C
        f.write(f"{-pixel_size}\n")  # D
        f.write(f"{west}\n")  # E
        f.write(f"{north}\n")  # F

    print("Saved world:", tfw_path)
    print("Top-left lon/lat:", west, north)

    return


# mt1.google.com、Mapbox、OpenStreetMap、Bing无偏移,避免使用国内图源
def download_single_tiles(west, south, east, north, zoom, source_type="google", output_dir="tiles"):
    """
    下载单个瓦片并保存为独立的 GeoTIFF (EPSG:3857)。

    参数:
        west, south, east, north: 目标区域经纬度 (WGS84)
        zoom: 缩放级别 (18=约0.6米精度, 19=约0.3米精度)
        output_dir: 保存目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 获取该区域覆盖的所有瓦片 (x, y, z)
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    total = len(tiles)
    print(f"区域内共有 {total} 个瓦片，准备下载...")

    # Google 卫星图源 (国内访问通常需要代理，或寻找国内镜像源)
    # 这里的 url 仅作示例，实际使用请确保网络通畅
    url_template = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

    if source_type == "arcgis":
        # ArcGIS 使用标准的 XYZ
        url_template = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    elif source_type == "bing":
        # Bing 需要计算 QuadKey
        url_template = "http://ecn.t1.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
    elif source_type == 'mapbox':
        url_template = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=YOUR_ACCESS_TOKEN'
    elif source_type == 'tianditu':
        # 虽然是官方地图，但在公共 Web 服务中，它的瓦片切割原点和坐标系（CGCS2000）在某些缩放级别下与标准的 Google Web Mercator 存在像素级错位，且现在需要强制申请 Key
        url = ("http://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default"
               "&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&tk={tk}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36"
    }

    success_count = 0

    for i, tile in enumerate(tiles):
        # 2. 构造请求
        if source_type == 'bing':
            quadkey = mercantile.quadkey(tile)
            url = url_template.format(q=quadkey)
        else:
            url = url_template.format(x=tile.x, y=tile.y, z=tile.z)

        file_name = f"tile_z{tile.z}_x{tile.x}_y{tile.y}.tif"
        save_path = os.path.join(output_dir, file_name)

        # 如果文件已存在，跳过 (支持断点续传)
        if os.path.exists(save_path):
            success_count += 1
            # print(f"[{i + 1}/{total}] 已存在: {file_name}")
            continue

        try:
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=10)
            if resp.status_code != 200:
                print(f"[{i + 1}/{total}] 下载失败 {resp.status_code}: {url}")
                continue

            # 3. 读取二进制图片数据
            # 使用 rasterio 的 MemoryFile 直接在内存处理，无需 PIL
            with rasterio.MemoryFile(resp.content) as memfile:
                with memfile.open() as dataset:
                    img_data = dataset.read()  # 读取数据 (Bands, H, W)

                    # 检查波段数 (有的瓦片可能是单通道或4通道)
                    count = dataset.count
                    height = dataset.height
                    width = dataset.width

            # 4. 核心：计算该瓦片的地理坐标范围 (Web Mercator, EPSG:3857)
            # mercantile.xy_bounds 返回单位是“米”，完美对应 EPSG:3857
            bounds = mercantile.xy_bounds(tile)
            west, south, east, north = bounds

            # 5. 生成仿射变换矩阵 (Affine Transform)
            # 这告诉 GIS 软件：图片左上角对应地图上的哪个坐标，每个像素代表多少米
            # 顺序固定: 西, 南, 东, 北, 宽, 高
            transform = from_bounds(west, south, east, north, width, height)

            # 6. 写入 GeoTIFF
            with rasterio.open(
                    save_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=count,
                    dtype=img_data.dtype,
                    crs='EPSG:3857',  # 关键：声明坐标系为 Web Mercator
                    transform=transform,  # 关键：写入坐标位置
                    compress='lzw'  # 使用无损压缩
            ) as dst:
                # img = np.array(img_data).transpose(2, 0, 1)  # RGB → CHW
                dst.write(img_data)

            print(f"[{i + 1}/{total}] 成功保存: {file_name}")
            success_count += 1

        except Exception as e:
            print(f"[{i + 1}/{total}] 处理出错: {e}")

    print(f"任务结束。成功: {success_count}/{total}")

    success = success_count == total
    return success


def download_one_tile(tile, url_template, output_dir, source_type, session):
    # 2. 构造请求
    if source_type == 'bing':
        quadkey = mercantile.quadkey(tile)
        url = url_template.format(q=quadkey)
    else:
        url = url_template.format(x=tile.x, y=tile.y, z=tile.z)

    file_name = f"tile_z{tile.z}_y{tile.y}_x{tile.x}.tif"  # 先行再列
    save_path = os.path.join(output_dir, file_name)

    # 如果文件已存在，跳过 (支持断点续传)
    if os.path.exists(save_path):
        # success_count += 1
        # print(f"[{i + 1}/{total}] 已存在: {file_name}")
        return f"Skipped {file_name}"

    try:
        # 使用传入的 session 发送请求 (复用 TCP 连接)
        resp = session.get(url, headers=headers, proxies=proxies, timeout=10)

        if resp.status_code != 200:
            # print(f"[{i + 1}/{total}] 下载失败 {resp.status_code}: {url}")
            print(f"下载失败 {resp.status_code}: {url}")
            return f"Failed {file_name}"

        # 3. 读取二进制图片数据
        # 使用 rasterio 的 MemoryFile 直接在内存处理，无需 PIL
        with rasterio.MemoryFile(resp.content) as memfile:
            with memfile.open() as dataset:
                img_data = dataset.read()  # 读取数据 (Bands, H, W)

                # 检查波段数 (有的瓦片可能是单通道或4通道)
                count = dataset.count
                height = dataset.height
                width = dataset.width

        # 4. 核心：计算该瓦片的地理坐标范围 (Web Mercator, EPSG:3857)
        # mercantile.xy_bounds 返回单位是“米”，完美对应 EPSG:3857
        bounds = mercantile.xy_bounds(tile)
        west, south, east, north = bounds

        # 5. 生成仿射变换矩阵 (Affine Transform)
        # 顺序固定: 西, 南, 东, 北, 宽, 高
        transform = from_bounds(west, south, east, north, width, height)

        # 6. 写入 GeoTIFF
        with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=img_data.dtype,
                crs='EPSG:3857',  # 关键：声明坐标系为 Web Mercator
                transform=transform,  # 关键：写入坐标位置
                compress='lzw'  # 使用无损压缩
        ) as dst:
            dst.write(img_data)

        # print(f"[{i + 1}/{total}] 成功保存: {file_name}")
        # success_count += 1
        return f"Success {file_name}"
    except Exception as e:
        # print(f"[{i + 1}/{total}] 处理出错: {e}")
        print(f"处理出错: {e}")
        return f"Error {file_name}"


def download_single_layer_multiThread(west, south, east, north, zoom, max_workers=16, source_type="google",
                                      output_dir="tiles"):
    """
    下载单个瓦片并保存为独立的 GeoTIFF (EPSG:3857)。

    参数:
        west, south, east, north: 目标区域经纬度 (WGS84)
        zoom: 缩放级别 (18=约0.6米精度, 19=约0.3米精度)
        output_dir: 保存目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 获取该区域覆盖的所有瓦片 (x, y, z)
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    total = len(tiles)
    print(f"区域内共有 {total} 个瓦片，准备下载...")

    url_template = get_url_template(source_type)

    # 2. 配置 Session (连接池 + 重试机制)
    # 这一步非常关键！高并发下必须复用连接，否则会耗尽本地端口
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # 3. 开启线程池
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(download_one_tile, t, url_template, output_dir, source_type, session):
                       t for t in tiles}

        # 处理结果 (进度条)
        for i, future in enumerate(as_completed(futures)):
            ret = future.result()
            result, file_name = ret.split()
            if i % 100 == 0:
                print(f"进度: {i}/{len(tiles)}")

            if result == "Success":
                print(f"[{i + 1}/{total}] 成功保存: {file_name}")
                success_count += 1

            if result == "Skipped":
                success_count += 1

    print(f"任务结束。成功: {success_count}/{total}")
    success = success_count == total

    return success


async def download_one_tile_async(tile, url_template, output_dir, source_type, session):
    # 2. 构造请求
    if source_type == 'bing':
        quadkey = mercantile.quadkey(tile)
        url = url_template.format(q=quadkey)
    else:
        url = url_template.format(x=tile.x, y=tile.y, z=tile.z)

    file_name = f"tile_z{tile.z}_y{tile.y}_x{tile.x}.tif"  # 先行再列
    save_path = os.path.join(output_dir, file_name)

    # 如果文件已存在，跳过 (支持断点续传)
    if os.path.exists(save_path):
        # success_count += 1
        # print(f"[{i + 1}/{total}] 已存在: {file_name}")
        return f"Skipped {file_name}"

    try:
        # 使用传入的 session 发送请求 (复用 TCP 连接)
        async with session.get(url, headers=headers, proxy=http_proxy, timeout=10) as resp:
            if resp.status == 200:
                content = await resp.read()

                async with rasterio.MemoryFile(resp.content) as memfile:
                    with memfile.open() as dataset:
                        img_data = dataset.read()  # 读取数据 (Bands, H, W)

                        # 检查波段数 (有的瓦片可能是单通道或4通道)
                        count = dataset.count
                        height = dataset.height
                        width = dataset.width

                    bounds = mercantile.xy_bounds(tile)
                    west, south, east, north = bounds
                    transform = from_bounds(west, south, east, north, width, height)

                # 异步写入文件，防止磁盘 IO 阻塞网络请求
                async with rasterio.open(
                        save_path,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=count,
                        dtype=img_data.dtype,
                        crs='EPSG:3857',  # 关键：声明坐标系为 Web Mercator
                        transform=transform,  # 关键：写入坐标位置
                        compress='lzw'  # 使用无损压缩
                ) as dst:
                    dst.write(img_data)

            else:
                # print(f"[{i + 1}/{total}] 下载失败 {resp.status_code}: {url}")
                print(f"下载失败 {resp.status_code}: {url}")
                return f"Failed {file_name}"

        # print(f"[{i + 1}/{total}] 成功保存: {file_name}")
        # success_count += 1
        return f"Success {file_name}"
    except Exception as e:
        # print(f"[{i + 1}/{total}] 处理出错: {e}")
        print(f"处理出错: {e}")
        return f"Error {file_name}"


async def download_single_layer_async(west, south, east, north, zoom, source_type="google", concurrency=50,
                                      output_dir="tiles"):
    """
    下载单个瓦片并保存为独立的 GeoTIFF (EPSG:3857)。

    参数:
        west, south, east, north: 目标区域经纬度 (WGS84)
        zoom: 缩放级别 (18=约0.6米精度, 19=约0.3米精度)
        output_dir: 保存目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    url_template = get_url_template(source_type)

    # 1. 获取该区域覆盖的所有瓦片 (x, y, z)
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    total = len(tiles)
    print(f"区域内共有 {total} 个瓦片，准备下载...")

    # 限制并发量 (Semaphore)，防止瞬间几万个请求把服务器打挂或被封 IP
    sem = asyncio.Semaphore(concurrency)

    async def bound_fetch(session, tile):
        async with sem:
            await download_one_tile_async(tile, url_template, output_dir, source_type, session)

    # 建立单一 TCP 连接池
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [bound_fetch(session, t) for t in tiles]
        # 使用 tqdm 显示进度条 (需要 pip install tqdm)
        from tqdm.asyncio import tqdm
        await tqdm.gather(*tasks)

    return True


@timer
def download_pyramid_imagery(max_times=5, mosaic=False):
    """
    常用层级速查表 (赤道基准)
    Zoom	分辨率 (米/像素)	描述
    15	4.777	城市街区
    16	2.388	大型建筑
    17	1.194	汽车轮廓
    18	0.597	Google/ArcGIS 常规高清
    19	0.298	亚米级 (极高清)
    20	0.149	往往是数字放大
    """
    save_path = 'pyramid_images'
    # save_path = '/media/fangjieyu/SU710/whu_project/Sat-VINS-ESEKF/dataset/pyramid_images'
    zoom_top = 12
    zoom_bottom = 18
    source_type = 'bing'
    bbox = (108.9, 18.7, 109.3, 18.84)
    bbox = (108.94, 18.76, 109.26, 18.8)
    bbox = (108.94, 18.75, 109.26, 18.8)  # 乐东 抱由镇 0.32,0.05
    bbox = (109.1, 34.58, 109.43, 34.75)  # 阎良区 0.33,0.17

    # 水经微注图层级索引是从1开始的
    resolutions = get_webmercator_resolutions()

    err_layers = []
    try_times = 0
    for i, zoom in enumerate(range(zoom_top, zoom_bottom + 1)):
        sub_dir = Path(save_path, f"L{zoom}").as_posix()
        res = resolutions[zoom]

        while try_times < max_times:
            try_times += 1
            print(f'\n第{try_times}次 trying to download L{zoom} {res:.2f}m ...')

            # 10分钟左右下完12-18级,L18 32*234 5*36km~210km2
            success = download_single_layer_multiThread(*bbox, zoom, source_type=source_type, output_dir=sub_dir)
            # success = download_single_tiles(*bbox, zoom, source_type, output_dir=sub_dir)

            if success:
                print(f'downloaded L{zoom} successfully.')
                if mosaic and zoom <= 16:
                    merged_tif = Path(save_path) / f'L{zoom}.tif'
                    if not merged_tif.exists():
                        mosaic_via_vrt(sub_dir, merged_tif.as_posix())
                        print(f'merged {merged_tif} successfully.')
                    else:
                        print(f'{merged_tif} exists, skip')

                try_times = 0
                break

        if try_times == max_times:
            # 5次依然没有全部下载成功
            err_layers.append(zoom)
            try_times = 0

    if len(err_layers) == 0:
        print('all layers downloaded successfully.')
    else:
        print(f'failed layers: {err_layers}')

    return


def merge_geotiff_directory(input_dir, output_tif):
    """
    拼接目录下所有 GeoTIFF 到一个大图
    """

    tif_list = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".tif")
    ]

    if len(tif_list) == 0:
        raise ValueError("目录中没有 .tif 文件")

    print("读取 TIFF 文件数量：", len(tif_list))

    datasets = []
    for tif in tif_list:
        src = rasterio.open(tif)
        datasets.append(src)
        print("加载：", tif)

    print("开始拼接 ...")
    mosaic, out_transform = merge(datasets)

    # 使用第一个影像的 profile 作为模板
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw",  # 可改为 deflate 或 jpeg
        "BIGTIFF": "IF_SAFER",  # >4GB 自动升级 BigTIFF格式
    })

    with rasterio.open(output_tif, "w", **out_meta) as dst:
        dst.write(mosaic)

    print("完成：", output_tif)

    return


def mosaic_tifs(input_dir, output_path):
    """
    将目录下的所有 GeoTIFF 拼接成一张大图
    """
    # 1. 获取所有 TIF 文件路径
    # 根据你的实际后缀调整，可能是 .tif 或 .tiff
    search_criteria = "*.tif"
    q = os.path.join(input_dir, search_criteria)
    file_paths = glob.glob(q)

    if not file_paths:
        print("未找到 TIFF 文件！")
        return

    print(f"找到 {len(file_paths)} 个文件，准备拼接...")

    # 2. 打开所有源文件
    # 注意：这里只是打开文件句柄，并没有一次性把所有像素读入内存
    src_files_to_mosaic = []
    try:
        for fp in file_paths:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # 3. 执行拼接 (Merge)
        # 这一步最耗内存，rasterio 会计算大图范围并将数据填入
        # mosaic: 拼接后的 numpy 数组 (bands, height, width)
        # out_trans: 新的仿射变换矩阵
        print("正在执行合并计算...")
        mosaic, out_trans = merge(src_files_to_mosaic)

        # 4. 复制并更新元数据
        # 使用第一个文件的元数据作为模板
        out_meta = src_files_to_mosaic[0].meta.copy()

        # 更新关键信息：高、宽、变换矩阵
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src_files_to_mosaic[0].crs,
            # === 关键优化 ===
            "compress": "lzw",  # 使用 LZW 无损压缩，大幅减小文件体积
            "tiled": True,  # 开启分块存储，读取更快
            "bigtiff": "IF_NEEDED"  # 如果文件超过 4GB，自动切换为 BigTIFF 格式
        })

        # 5. 写入输出文件
        print(f"正在写入文件: {output_path} ...")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print("拼接成功！")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 6. 关闭所有文件句柄
        for src in src_files_to_mosaic:
            src.close()


@timer
def mosaic_via_vrt(input_dir, output_path):
    '''
    sudo apt update
    sudo apt install gdal-bin libgdal-dev python3-gdal

    conda install -c conda-forge gdal # 网不好时也可以whl安装
    pip install gdal==3.4.1 # recommend

    gdalinfo --version
    gdal-config --version
    python3 -c "from osgeo import gdal; print(gdal.__version__)"

    gdalbuildvrt mosaic.vrt *.tif
    gdal_translate mosaic.vrt merged.tif -co COMPRESS=LZW

    subprocess.run("gdalbuildvrt mosaic.vrt *.tif", shell=True)
    subprocess.run("gdal_translate mosaic.vrt merged.tif -co COMPRESS=LZW", shell=True)
    '''

    from osgeo import gdal

    # 1. 获取文件列表
    files = glob.glob(os.path.join(input_dir, "*.tif"))

    # 2. 构建 VRT (虚拟文件)
    # 这一步非常快，因为不移动像素
    vrt_path = output_path.replace(".tif", ".vrt")
    # print("构建虚拟索引 VRT...")
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=True)
    gdal.BuildVRT(vrt_path, files, options=vrt_options)

    # 3. (可选) 将 VRT 转为真实的 TIFF
    # 如果你只需要在 GIS 软件里看，直接拖入 VRT 文件即可，不需要这一步
    # 如果必须生成实体 TIFF 文件，执行 Translate
    # print("转换实体 TIFF (这步需要时间)...")
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_NEEDED"]
    )
    gdal.Translate(output_path, vrt_path, options=translate_options)

    # 清理中间文件
    if os.path.exists(vrt_path):
        os.remove(vrt_path)
    # print("完成！")


def get_webmercator_resolutions(max_zoom=21):
    resolutions = {}
    for z in range(max_zoom + 1):
        res = 156543.03392804097 / (2 ** z)  # meters per pixel
        # res_ = res * math.cos(math.radians(lat))
        resolutions[z] = res
    return resolutions


def calculate_web_mercator_resolutions(max_zoom=23):
    """
    计算 Google/Bing/ArcGIS 标准切片方案的理论分辨率
    基于 EPSG:3857 投影，Tile Size = 256x256
    """
    # 地球赤道周长 (米) - WGS84 标准
    # 2 * pi * 6378137
    earth_circumference = 40075016.686

    print(f"{'Zoom':<5} | {'Resolution (m/pixel)':<25} | {'Scale (approx)':<20}")
    print("-" * 55)

    resolutions = {}

    for z in range(max_zoom + 1):
        # 核心公式: 周长 / (256 * 2^z)
        resolution = earth_circumference / (256 * (2 ** z))

        # 估算比例尺 (假设屏幕 PPI 为 96)
        scale = resolution * 96 / 0.0254

        resolutions[z] = resolution

        # 打印格式化信息
        # Zoom 18 约 0.6米, Zoom 19 约 0.3米
        print(f"{z:<5} | {resolution:<25.9f} | 1 : {int(scale):,}")

    return resolutions


def test():
    download_satellite_imagery('arcgis')

    # 示例：北京天安门附近一个小区域
    # (西经, 南纬, 东经, 北纬)
    bbox = (116.388, 39.903, 116.398, 39.910)
    bbox = (108.9, 18.7, 109.3, 18.84)

    # Zoom 19 约为 0.3米/像素 (亚米级)
    # Zoom 18 约为 0.6米/像素
    # download_single_tiles(*bbox, zoom=14, source_type='arcgis')
    # merge_geotiff_directory('tiles', 'merged.tif')
    mosaic_via_vrt('tiles', 'merged.tif')


if __name__ == '__main__':
    # test()
    # download_pyramid_imagery(mosaic=True)
    download_pyramid_imagery(mosaic=True)
    pass
