import socket
import json
import cv2
import random
import numpy as np
import pymap3d as pm

# !!! 修改成 Ubuntu 服务器的局域网 IP !!!
SERVER_IP = "192.168.31.75"
SERVER_PORT = 50007

random.seed(0)  # 可去掉种子以获得每次不同的值


def enu_basis_from_latlon(lat, lon):
    """
    返回局部 ENU 基向量（每列为一个基向量）在 ECEF 坐标系下的表示：
      E = basis[:,0], N = basis[:,1], U = basis[:,2]
    所以 ECEF_vec = E*e + N*n + U*u
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # 东、北、上 向量（每个都是长度为3的列向量）
    E = np.array([-sin_lon, cos_lon, 0.0])
    N = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    U = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    basis = np.column_stack((E, N, U))
    return basis


def relative_to_ecef(A_ecef, r, az_deg, el_deg, az_unit='deg', el_unit='deg'):
    """
    A_ecef: iterable-like [xA, yA, zA] (meters)
    r: 距离（米）
    az_deg: 水平方位角（从北顺时针），单位deg或rad，见 az_unit
    el_deg: 仰角（相对于水平面，向上为正），单位deg或rad，见 el_unit
    返回 B_ecef (x,y,z) in meters
    """
    xA, yA, zA = A_ecef
    if az_unit == 'deg':
        az = np.deg2rad(az_deg)
    else:
        az = az_deg
    if el_unit == 'deg':
        el = np.deg2rad(el_deg)
    else:
        el = el_deg

    # 1) 求 A 的经纬度
    lat, lon, _h = pm.ecef2geodetic(xA, yA, zA)

    # 2) 建立 ENU 基向量（在 ECEF 表示）
    basis = enu_basis_from_latlon(lat, lon)  # columns: E, N, U

    # 3) 在 ENU 中构造向量分量（注意轴顺序：E, N, U）
    # az: 从北（N）顺时针，el: 仰角
    e_comp = r * np.cos(el) * np.sin(az)
    n_comp = r * np.cos(el) * np.cos(az)
    u_comp = r * np.sin(el)
    enu_vec = np.array([e_comp, n_comp, u_comp])

    # 4) 转换到 ECEF
    delta_ecef = basis.dot(enu_vec)  # ECEF 增量
    B_ecef = np.array([xA, yA, zA]) + delta_ecef
    return B_ecef


def main():
    # 连接服务器
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, SERVER_PORT))
    print(f"[CLIENT] Connected to {SERVER_IP}:{SERVER_PORT}")

    init = False
    lat0, lon0 = 0, 0

    buffer = ""
    while True:
        # 接收数据
        data = client_socket.recv(4096).decode("utf-8")
        if not data:
            break

        d = random.uniform(600, 800)
        alpha = random.uniform(110, 120)
        beta = random.uniform(-10, 10)

        buffer += data
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            try:
                msg = json.loads(line)
                pos = msg["position"]
                pos1 = relative_to_ecef(pos, d, alpha, beta)  # 相对导航

                if not init:
                    lat0, lon0, alt0 = pm.ecef2geodetic(pos[0], pos[1], pos[2])
                    print(f"[CLIENT] Initial position: {lat0}, {lon0}, {alt0}")
                    init = True

                lat, lon, alt = pm.ecef2geodetic(pos[0], pos[1], pos[2])
                dlat = lat - lat0
                dlon = lon - lon0

                lat1, lon1, alt1 = pm.ecef2geodetic(pos1[0], pos1[1], pos1[2])
                dlat1 = lat1 - lat0
                dlon1 = lon1 - lon0

                # 生成可视化画面
                img = np.zeros((500, 500, 3), dtype=np.uint8)
                center = (250, 250)
                coef = 1e4
                point = (int(center[0] + dlon * coef), int(center[1] - dlat * coef))
                point1 = (int(center[0] + dlon1 * coef), int(center[1] - dlat1 * coef))
                print(f"[CLIENT] Lat: {lat}, Lon: {lon}, Alt: {alt}")

                cv2.circle(img, point, 5, (0, 0, 255), -1)
                cv2.circle(img, point1, 4, (255, 0, 0), -1)
                cv2.putText(img, f"X:{lat:.2f} Y:{lon:.2f} Z:{alt:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Position Visualization", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
            except json.JSONDecodeError:
                pass


if __name__ == "__main__":
    main()
