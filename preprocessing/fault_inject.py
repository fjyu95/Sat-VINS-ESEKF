"""故障/错误数据注入工具（文本类数据文件）。

对数值文本（航拍元数据 txt、轨迹 csv 等）注入三种简单故障：
- bias：比例偏置（数值 × (1 + 强度%)）
- noise：随机噪声（数值 × (1 + N(0, 强度%))）
- drop：数据中断（数值置 0）

键值对行（"key: value"）只扰动带物理单位（度/米/秒）的数值，
避免破坏时间戳等非物理字段；纯数值行（csv/空白分隔数据）全部数值都注入。
"""
import re
from pathlib import Path

import numpy as np

_NUM = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
_NUMERIC_HINTS = ('度)', '米)', '秒)')


def corrupt_text(text, kind, strength, rng=None):
    """逐行扫描数值并注入故障，返回 (新文本, 修改的数值个数)。"""
    rng = rng or np.random.default_rng()
    n_changed = 0

    def _sub(m):
        nonlocal n_changed
        v = float(m.group())
        if kind == 'bias':
            v *= 1.0 + strength / 100.0
        elif kind == 'noise':
            v *= 1.0 + rng.normal(0.0, strength / 100.0)
        else:                      # drop：数据中断，置零
            v = 0.0
        n_changed += 1
        return f'{v:.6f}'

    lines = []
    for line in text.splitlines():
        if ':' in line and not any(h in line for h in _NUMERIC_HINTS):
            lines.append(line)     # 非物理字段（时间戳/名称等）原样保留
            continue
        lines.append(_NUM.sub(_sub, line))
    return '\n'.join(lines) + '\n', n_changed


def inject_file(path, kind, strength, rng=None):
    """对文件注入故障：备份为 *.bak 后覆盖写回原路径。

    返回 (修改的数值个数, 备份文件路径)。
    """
    path = Path(path)
    raw = path.read_bytes()
    for enc in ('gb18030', 'utf-8'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode('utf-8', errors='replace')
        enc = 'utf-8'

    new_text, n_changed = corrupt_text(text, kind, strength, rng)

    bak = path.with_suffix(path.suffix + '.bak')
    if not bak.exists():
        bak.write_bytes(raw)
    path.write_bytes(new_text.encode(enc, errors='replace'))
    return n_changed, bak


def restore_file(path):
    """从 *.bak 备份恢复原文件，返回是否成功。"""
    path = Path(path)
    bak = path.with_suffix(path.suffix + '.bak')
    if not bak.exists():
        return False
    path.write_bytes(bak.read_bytes())
    bak.unlink()
    return True
