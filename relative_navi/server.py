import socket
import time
import json
import random

import numpy as np
import pymap3d as pm

HOST = "0.0.0.0"  # 监听所有网卡
PORT = 50007  # 自定义端口

enu_datum_file = 'enu_datum.txt'
filepath = 'traj_esekf_out_f1.txt'  # timestamp x y z qx qy qz qw

with  open(enu_datum_file, 'r') as f:
    enu_datum = np.array([float(x) for x in f.readline().split()])
arr = np.loadtxt(filepath, delimiter=' ')
n_samples = arr.shape[0]


def enu_to_ecef(xyz, enu):
    x, y, z = xyz
    lat0, lon0, h0 = enu
    xyz_ecef = pm.enu2ecef(x, y, z, lat0, lon0, h0)
    return xyz_ecef


def main():
    # 创建 TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[SERVER] Listening on {HOST}:{PORT}")

    conn, addr = server_socket.accept()
    print(f"[SERVER] Connected by {addr}")

    try:
        i = 0
        while True:
            # 飞行定位数据 (x, y, z)
            xyz = arr[i % n_samples][1:4]
            xyz_ecef = enu_to_ecef(xyz, enu_datum)
            data = {
                "timestamp": time.time(),
                "position": xyz_ecef
            }
            json_str = json.dumps(data)
            print(f"[SERVER] Sending data: {json_str}")

            conn.sendall((json_str + "\n").encode("utf-8"))
            time.sleep(0.5)  # 0.5 秒发送一次
            i += 1

    except (ConnectionResetError, BrokenPipeError):
        print("[SERVER] Client disconnected")
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SERVER] Ctrl+C 退出")
