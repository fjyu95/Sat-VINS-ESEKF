import socket
import time
import json
import threading

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


def client_worker(conn, addr):
    """单个从机连接的发送线程：0.5 s 周期广播主机轨迹流。
    多从机同时连接互不干扰，断开时仅清理本连接。"""
    i = 0
    try:
        while True:
            # 飞行定位数据 (x, y, z)
            xyz = arr[i % n_samples][1:4]
            xyz_ecef = enu_to_ecef(xyz, enu_datum)
            data = {
                "timestamp": time.time(),
                "position": xyz_ecef
            }
            conn.sendall((json.dumps(data) + "\n").encode("utf-8"))
            time.sleep(0.5)  # 0.5 秒发送一次
            i += 1

    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        conn.close()
        print(f"[SERVER] 从机断开: {addr}")


def main():
    # 创建 TCP socket，支持多台从机并发连接（1 主机 + N 从机）
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(8)
    print(f"[SERVER] Listening on {HOST}:{PORT}  (多从机模式, Ctrl+C 退出)")

    n_client = 0
    while True:
        conn, addr = server_socket.accept()
        n_client += 1
        print(f"[SERVER] WING-{n_client} 接入 {addr}")
        threading.Thread(target=client_worker,
                         args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SERVER] Ctrl+C 退出")
