# -*- coding: utf-8 -*-
# 临时补丁脚本：为 relative_navi/client.py 加入多从机支持，用后立即删除
p = 'relative_navi/client.py'
s = open(p, encoding='utf-8').read()
edits = [
    ('import socket\nimport json',
     'import socket\nimport sys\nimport json'),
    ('random.seed(0)  # 可去掉种子以获得每次不同的值',
     'CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 僚机编号: python client.py <id>\nrandom.seed(CLIENT_ID)'),
    ('    def __init__(self):',
     '    def __init__(self, wid=0):\n        self.wid = wid'),
    ('("WINGMAN", wing_geo, C_WING)',
     '(f"WING-{self.wid}", wing_geo, C_WING)'),
    ('put_text(img, wing_pt[0] + 14, wing_pt[1] - 12, "WING", 0.42, C_WING)',
     'put_text(img, wing_pt[0] + 14, wing_pt[1] - 12, f"WING-{self.wid}", 0.42, C_WING)'),
    ('hud = HudRenderer()',
     'hud = HudRenderer(CLIENT_ID)'),
    ('cv2.imshow("GeoVIO Relative Navigation", frame)',
     'cv2.imshow(f"GeoVIO Relative Navigation WING-{CLIENT_ID}", frame)'),
]
for old, new in edits:
    assert old in s, f'pattern not found: {old[:40]}'
    s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('patched', len(edits), 'edits')
