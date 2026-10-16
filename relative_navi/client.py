import socket
import sys
import json
import random
from collections import deque
from math import cos, sin, radians, hypot, pi

import cv2
import numpy as np
import pymap3d as pm

# !!! 修改成 Ubuntu 服务器的局域网 IP !!!
# SERVER_IP = "127.0.0.1" # 单机测试
SERVER_IP = "192.168.31.75"
SERVER_PORT = 50007

# 僚机编号：python client.py <id>。多从机部署时每台从机使用不同 id，
# 以 id 作随机种子生成各自独立的相对位置（同 id 可复现同序列）
CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(CLIENT_ID)

# ------------------------------------------------------------------
# 可视化配色（BGR）
# ------------------------------------------------------------------
C_BG    = (34, 20, 10)    # 深空蓝背景
C_PANEL = (46, 28, 14)    # 面板底色
C_RING  = (96, 74, 44)    # 距离环
C_GRID  = (70, 54, 32)    # 网格/分隔线
C_SWEEP = (60, 140, 60)   # 雷达扫描线
C_TEXT  = (226, 232, 240) # 主文字
C_DIM   = (138, 150, 162) # 次级文字
C_LEAD  = (78, 104, 252)  # 长机（暖橙红）
C_WING  = (238, 205, 90)  # 僚机（青蓝）
C_LINK  = (80, 200, 130)  # 链路/状态绿
C_AMBER = (60, 190, 255)  # 琥珀色强调

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTD = cv2.FONT_HERSHEY_DUPLEX


def put_text(img, x, y, text, scale=0.5, color=C_TEXT, thick=1):
    cv2.putText(img, text, (int(x), int(y)), FONTD, scale, color, thick, cv2.LINE_AA)


def put_kv(img, x, y, w, label, value, unit="", scale=0.42):
    """标签左对齐 + 数值右对齐的键值行。"""
    put_text(img, x, y, label, scale, C_DIM)
    txt = f"{value}{unit}"
    tw = cv2.getTextSize(txt, FONTD, scale, 1)[0][0]
    put_text(img, x + w - tw, y, txt, scale, C_TEXT)


def blend_rect(img, x, y, w, h, color, alpha=0.6):
    roi = img[y:y + h, x:x + w]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_glow(img, pt, color, core=4):
    """带辉光效果的标记点。"""
    glow = np.zeros_like(img)
    cv2.circle(glow, pt, core * 4, color, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), core * 1.8)
    cv2.add(img, glow, dst=img)
    cv2.circle(img, pt, core, color, -1, cv2.LINE_AA)
    cv2.circle(img, pt, core + 3, color, 1, cv2.LINE_AA)


def draw_dashed(img, p1, p2, color, dash=8, gap=6, thick=1):
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1:
        return
    n = int(dist / (dash + gap)) + 1
    for i in range(n):
        a = p1 + (p2 - p1) * (i * (dash + gap)) / dist
        b = p1 + (p2 - p1) * min(i * (dash + gap) + dash, dist) / dist
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def draw_craft(img, pt, color, heading_deg, s=10):
    """三角机标：heading 0 指向北方（屏幕上方）。"""
    pts = []
    for d in (0, 135, -135):
        b = radians(heading_deg + d)
        pts.append([int(pt[0] + s * sin(b)), int(pt[1] - s * cos(b))])
    cv2.fillPoly(img, [np.array(pts)], color, cv2.LINE_AA)
    draw_glow(img, pt, color, core=3)


class HudRenderer:
    """雷达态势图 + 遥测面板渲染器（长机居中，僚机显示相对位置）。"""

    W, H = 1080, 720
    PANEL_X = 780                 # 右侧遥测面板起始 x
    CX, CY, R = 390, 390, 300     # 雷达圆心与半径（px）

    def __init__(self, wid=0):
        self.wid = wid                  # 僚机编号
        self.trail = deque(maxlen=160)   # 僚机相对轨迹（ENU, m）
        self.sweep_deg = 0.0
        self.count = 0

    # --------------------------------------------------------------
    def _base(self):
        img = np.full((self.H, self.W, 3), C_BG, np.uint8)
        blend_rect(img, self.PANEL_X, 0, self.W - self.PANEL_X, self.H, C_PANEL, 0.55)
        cv2.line(img, (self.PANEL_X, 0), (self.PANEL_X, self.H), C_GRID, 1, cv2.LINE_AA)
        # 标题
        put_text(img, 30, 42, "RELATIVE NAVIGATION SCOPE", 0.6, C_TEXT)
        put_text(img, 30, 62, "LOCAL ENU FRAME / PLAN VIEW", 0.36, C_DIM)
        return img

    def _scope(self, img, scope_rng):
        cx, cy, r = self.CX, self.CY, self.R
        # 十字基准线
        cv2.line(img, (cx - r, cy), (cx + r, cy), C_GRID, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r), (cx, cy + r), C_GRID, 1, cv2.LINE_AA)
        # 距离环 + 量程标注
        for k in (1, 2, 3):
            rr = int(r * k / 3)
            cv2.circle(img, (cx, cy), rr, C_RING, 1, cv2.LINE_AA)
            put_text(img, cx + rr * 0.72, cy - rr * 0.70,
                     f"{scope_rng * k / 3:.0f} m", 0.36, C_DIM)
        # 外圈刻度（15° 间隔，90° 加亮）
        for deg in range(0, 360, 15):
            a = radians(deg)
            x1 = cx + (r - 6) * sin(a); y1 = cy - (r - 6) * cos(a)
            x2 = cx + r * sin(a);       y2 = cy - r * cos(a)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                     C_TEXT if deg % 90 == 0 else C_GRID, 1, cv2.LINE_AA)
        # 方位字母
        for deg, ch in ((0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')):
            a = radians(deg)
            put_text(img, cx + (r + 18) * sin(a) - 5, cy - (r + 18) * cos(a) + 5,
                     ch, 0.5, C_AMBER if ch == 'N' else C_DIM)
        # 旋转扫描线
        a = radians(self.sweep_deg)
        p2 = (int(cx + r * sin(a)), int(cy - r * cos(a)))
        glow = np.zeros_like(img)
        cv2.line(glow, (cx, cy), p2, C_SWEEP, 1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 4)
        cv2.add(img, glow, dst=img)
        self.sweep_deg = (self.sweep_deg + 1.2) % 360

    def _hud(self, img, lead_geo, wing_geo, rel, rng, az, el, stamp):
        x0, w = self.PANEL_X + 24, self.W - self.PANEL_X - 48
        y = 44
        put_text(img, x0, y, "GEOVIO / REL NAV", 0.62, C_TEXT)
        put_text(img, x0, y + 22, "LEADER-FOLLOWER TELEMETRY", 0.34, C_DIM)

        # 链路状态
        draw_glow(img, (x0 + 5, y + 52), C_LINK, core=4)
        put_text(img, x0 + 16, y + 57, f"LINK OK   FRM {self.count}", 0.42, C_LINK)

        y += 92
        for name, geo, c in (("LEADER", lead_geo, C_LEAD),
                             (f"WING-{self.wid}", wing_geo, C_WING)):
            cv2.line(img, (x0, y - 8), (x0 + w, y - 8), C_GRID, 1, cv2.LINE_AA)
            put_text(img, x0, y + 12, name, 0.5, c)
            y += 40
            put_kv(img, x0, y, w, "LAT", f"{geo[0]:9.6f}", " deg"); y += 22
            put_kv(img, x0, y, w, "LON", f"{geo[1]:9.6f}", " deg"); y += 22
            put_kv(img, x0, y, w, "ALT", f"{geo[2]:8.2f}", " m");   y += 34

        cv2.line(img, (x0, y - 8), (x0 + w, y - 8), C_GRID, 1, cv2.LINE_AA)
        put_text(img, x0, y + 12, "RELATIVE", 0.5, C_AMBER)
        y += 40
        put_kv(img, x0, y, w, "RANGE",  f"{rng:8.2f}", " m");    y += 22
        put_kv(img, x0, y, w, "AZIMUTH", f"{az:7.2f}", " deg");  y += 22
        put_kv(img, x0, y, w, "ELEV",   f"{el:7.2f}", " deg");  y += 22
        put_kv(img, x0, y, w, "dE", f"{rel[0]:8.2f}", " m");     y += 22
        put_kv(img, x0, y, w, "dN", f"{rel[1]:8.2f}", " m");     y += 22
        put_kv(img, x0, y, w, "dU", f"{rel[2]:8.2f}", " m")

        put_text(img, x0, self.H - 40, f"TS {stamp:.3f}", 0.36, C_DIM)
        put_text(img, x0, self.H - 20, "[Q] EXIT", 0.36, C_DIM)

    # --------------------------------------------------------------
    def render(self, lead_geo, wing_geo, lead_enu, wing_enu, az, el, stamp):
        img = self._base()
        self.count += 1

        # 僚机相对长机的 ENU 偏移，长机居中显示
        rel = np.asarray(wing_enu, float) - np.asarray(lead_enu, float)
        rng3d = float(np.linalg.norm(rel))
        ground = hypot(rel[0], rel[1])
        scope_rng = max(ground, 30.0) * 1.3          # 自适应量程
        scale = self.R / scope_rng                    # px / m

        def to_px(e, n):
            return (int(self.CX + e * scale), int(self.CY - n * scale))

        self._scope(img, scope_rng)
        self.trail.append((rel[0], rel[1]))

        # 僚机相对轨迹（渐隐折线）
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = tuple(int(c * (0.25 + 0.75 * a)) for c in C_WING)
            cv2.line(img, to_px(*pts[i - 1]), to_px(*pts[i]), col, 2, cv2.LINE_AA)

        center = (self.CX, self.CY)
        wing_pt = to_px(rel[0], rel[1])

        # 长-僚连线 + 距离标签
        draw_dashed(img, center, wing_pt, C_LINK)
        mx, my = (center[0] + wing_pt[0]) // 2, (center[1] + wing_pt[1]) // 2
        label = f"{rng3d:.0f} m"
        tw = cv2.getTextSize(label, FONTD, 0.42, 1)[0][0]
        blend_rect(img, mx - tw // 2 - 6, my - 14, tw + 12, 20, C_PANEL, 0.75)
        put_text(img, mx - tw // 2, my + 1, label, 0.42, C_TEXT)

        # 机体标记
        draw_craft(img, center, C_LEAD, 0.0)
        draw_craft(img, wing_pt, C_WING, az)
        put_text(img, center[0] + 14, center[1] - 12, "LEAD", 0.42, C_LEAD)
        put_text(img, wing_pt[0] + 14, wing_pt[1] - 12,
                 f"WING-{self.wid}", 0.42, C_WING)

        self._hud(img, lead_geo, wing_geo, rel, rng3d, az, el, stamp)
        return img


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
    try:
        client_socket.connect((SERVER_IP, SERVER_PORT))
    except OSError as e:
        print(f"[CLIENT] 无法连接 {SERVER_IP}:{SERVER_PORT} -> {e}")
        return
    print(f"[CLIENT] Connected to {SERVER_IP}:{SERVER_PORT}")

    hud = HudRenderer(CLIENT_ID)
    init = False
    gui_ok = True
    lat0 = lon0 = alt0 = 0.0

    buffer = ""
    while True:
        # 接收数据
        try:
            data = client_socket.recv(4096)
        except OSError:
            break
        if not data:
            break
        data = data.decode("utf-8")

        d = random.uniform(600, 800)
        alpha = random.uniform(110, 120)
        beta = random.uniform(-10, 10)

        buffer += data
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            pos = msg["position"]
            pos1 = relative_to_ecef(pos, d, alpha, beta)  # 相对导航

            if not init:
                lat0, lon0, alt0 = pm.ecef2geodetic(*pos)
                print(f"[CLIENT] Initial position: {lat0:.6f}, {lon0:.6f}, {alt0:.2f}")
                init = True

            # 首帧为原点的局部 ENU 坐标
            lead_geo = pm.ecef2geodetic(*pos)
            wing_geo = pm.ecef2geodetic(*pos1)
            lead_enu = pm.ecef2enu(*pos, lat0, lon0, alt0)
            wing_enu = pm.ecef2enu(*pos1, lat0, lon0, alt0)

            print(f"[CLIENT] Leader lat={lead_geo[0]:.6f} "
                  f"lon={lead_geo[1]:.6f} alt={lead_geo[2]:.2f}")

            frame = hud.render(lead_geo, wing_geo, lead_enu, wing_enu,
                               az=alpha, el=beta, stamp=msg.get("timestamp", 0))
            if gui_ok:
                try:
                    cv2.imshow(f"GeoVIO Relative Navigation WING-{CLIENT_ID}", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    gui_ok = False
                    print("[CLIENT] OpenCV 无 GUI 支持（安装了 headless 版），"
                          "仅打印日志。修复：pip uninstall opencv-python-headless 后 "
                          "pip install opencv-python")

    client_socket.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CLIENT] Ctrl+C 退出")
