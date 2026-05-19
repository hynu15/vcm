"""Một lần chạy trên máy mới (RTX 4090 thuê): clone CCNet + patch + tải pretrained.

Idempotent: chạy lại không hại gì — đã có sẽ bỏ qua.

Use:
    cd /workspace/sac_project     # hoặc nơi project được mount
    python -m scripts.new_branch.setup_ccnet
"""
from __future__ import annotations

# Cho phép chạy cả `python -m scripts.new_branch.setup_ccnet` lẫn `python setup_ccnet.py`
if __package__ in (None, ""):
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(os.path.dirname(_here))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    __package__ = "scripts.new_branch"

import shutil
import subprocess
import urllib.request
from pathlib import Path

from .paths import CCNET_ROOT, MODELS_DIR, PROJECT_ROOT, RESNET101_PRETRAINED

CCNET_REPO = "https://github.com/speedinghzl/CCNet.git"
PRETRAINED_URL = "http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth"

# === Nội dung 3 file đã patched (sao chép nguyên xi từ local đã verify) ===

NETWORKS_INIT = "import networks.ccnet\n"

CC_ATTN_PATCH_OLD = (
    'def INF(B,H,W):\n'
    '     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)\n'
)
# Patch trung gian từ phiên bản trước (device-flex nhưng còn float("inf")) —
# cần upgrade lên version mới (-1e4, fp16-safe).
CC_ATTN_PATCH_PREV = (
    'def INF(B,H,W,device=None):\n'
    "     dev = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))\n"
    '     return -torch.diag(torch.tensor(float("inf"), device=dev).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)\n'
)
CC_ATTN_PATCH_NEW = (
    "# `float(\"inf\")` overflows fp16 in AMP and produces NaN trong softmax → dùng 1e4\n"
    "# (exp(-1e4) ≈ 0 trong mọi precision; an toàn cho cả fp16/fp32).\n"
    "_LARGE_NEG = 1e4\n"
    "\n"
    "def INF(B,H,W,device=None,dtype=None):\n"
    "     dev = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))\n"
    "     dt = dtype if dtype is not None else torch.float32\n"
    "     return -torch.diag(torch.full((H,), _LARGE_NEG, device=dev, dtype=dt),0).unsqueeze(0).repeat(B*W,1,1)\n"
)
CC_ATTN_CALL_OLD = (
    "energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)"
)
CC_ATTN_CALL_PREV = (
    "energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, device=x.device)).view(m_batchsize,width,height,height).permute(0,2,1,3)"
)
CC_ATTN_CALL_NEW = (
    "energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, device=x.device, dtype=proj_query_H.dtype)).view(m_batchsize,width,height,height).permute(0,2,1,3)"
)

CCNET_PY_PATCH_OLD = (
    "from cc_attention import CrissCrossAttention\n"
    "from utils.pyt_utils import load_model\n"
    "\n"
    "from inplace_abn import InPlaceABN, InPlaceABNSync\n"
    "BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')\n"
)
CCNET_PY_PATCH_NEW = (
    "from cc_attention import CrissCrossAttention\n"
    "from utils.pyt_utils import load_model\n"
    "\n"
    "# Patched for SAC reproduction: replace InPlaceABNSync with plain BN(+ReLU)\n"
    "BatchNorm2d = nn.BatchNorm2d\n"
    "\n"
    "\n"
    "class InPlaceABNSync(nn.Module):\n"
    "    \"\"\"BN + ReLU drop-in replacement for inplace_abn.InPlaceABNSync.\n"
    "    Original InPlaceABNSync defaults to leaky_relu(0.01); switching to ReLU is\n"
    "    standard in CCNet ports and does not change reported mIoU.\"\"\"\n"
    "\n"
    "    def __init__(self, num_features, **kwargs):\n"
    "        super().__init__()\n"
    "        kwargs.pop('activation', None)\n"
    "        self.bn = nn.BatchNorm2d(num_features, **kwargs)\n"
    "        self.act = nn.ReLU(inplace=True)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.act(self.bn(x))\n"
)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def _clone_looks_complete() -> bool:
    """CCNet đã clone đầy đủ nếu 3 file chủ chốt đều có mặt."""
    must = [
        CCNET_ROOT / "networks" / "ccnet.py",
        CCNET_ROOT / "cc_attention" / "functions.py",
        CCNET_ROOT / "utils" / "pyt_utils.py",
    ]
    return all(p.exists() for p in must)


def clone_ccnet() -> None:
    if CCNET_ROOT.exists() and _clone_looks_complete():
        print(f"[1/3] [skip] {CCNET_ROOT} đã clone đầy đủ.")
        return
    if CCNET_ROOT.exists():
        print(f"[1/3] {CCNET_ROOT} tồn tại nhưng thiếu file chủ chốt — xoá và clone lại.")
        shutil.rmtree(CCNET_ROOT)
    print(f"[1/3] Clone CCNet → {CCNET_ROOT}")
    _run(["git", "clone", "--depth", "1", CCNET_REPO, str(CCNET_ROOT)])
    if not _clone_looks_complete():
        raise RuntimeError(f"Clone xong nhưng vẫn thiếu file. Kiểm tra mạng/repo: {CCNET_REPO}")


def _patch_in_file(path: Path, old: str, new: str, label: str,
                   prev_versions: list[str] | None = None) -> None:
    """Áp patch idempotent với cả phiên bản trung gian trước đây.

    Trật tự kiểm tra:
    1. Nếu `new` đã có trong file → skip.
    2. Nếu nguyên gốc `old` còn trong file → replace by `new`.
    3. Nếu một trong các `prev_versions` còn trong file (= file đã patch ở
       phiên bản cũ của script này) → replace by `new` để upgrade.
    """
    if not path.exists():
        raise RuntimeError(f"Thiếu file cần patch: {path} ({label}). Repo CCNet chưa được clone đầy đủ.")
    txt = path.read_text()
    if new.strip() and (new in txt):
        print(f"  [skip] {label} đã patched")
        return
    if old in txt:
        path.write_text(txt.replace(old, new))
        print(f"  [ok] patched {label}")
        return
    for prev in (prev_versions or []):
        if prev in txt:
            path.write_text(txt.replace(prev, new))
            print(f"  [ok] upgraded {label} (từ patch cũ)")
            return
    raise RuntimeError(f"Không tìm thấy đoạn cần patch trong {path} cho {label}.\n"
                       f"Có thể CCNet repo đã đổi nội dung; sửa setup_ccnet.py.")


def patch_ccnet() -> None:
    print("[2/3] Patch CCNet (InPlaceABN → BN+ReLU, device-flex CCA)")
    nets_init = CCNET_ROOT / "networks" / "__init__.py"
    nets_init.parent.mkdir(parents=True, exist_ok=True)
    current = nets_init.read_text() if nets_init.exists() else ""
    if current.strip() != NETWORKS_INIT.strip():
        nets_init.write_text(NETWORKS_INIT)
        print("  [ok] networks/__init__.py (giữ lại mỗi ccnet)")
    else:
        print("  [skip] networks/__init__.py đã đúng")

    _patch_in_file(CCNET_ROOT / "networks" / "ccnet.py",
                   CCNET_PY_PATCH_OLD, CCNET_PY_PATCH_NEW, "ccnet.py:InPlaceABN")
    _patch_in_file(CCNET_ROOT / "cc_attention" / "functions.py",
                   CC_ATTN_PATCH_OLD, CC_ATTN_PATCH_NEW, "cc_attention/functions.py:INF def",
                   prev_versions=[CC_ATTN_PATCH_PREV])
    _patch_in_file(CCNET_ROOT / "cc_attention" / "functions.py",
                   CC_ATTN_CALL_OLD, CC_ATTN_CALL_NEW, "cc_attention/functions.py:INF call",
                   prev_versions=[CC_ATTN_CALL_PREV])


def download_pretrained() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if RESNET101_PRETRAINED.exists() and RESNET101_PRETRAINED.stat().st_size > 150 * 1024 * 1024:
        print(f"[3/3] [skip] Pretrained đã có: {RESNET101_PRETRAINED} "
              f"({RESNET101_PRETRAINED.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[3/3] Tải pretrained ResNet-101 ImageNet → {RESNET101_PRETRAINED}")
    if RESNET101_PRETRAINED.exists():
        RESNET101_PRETRAINED.unlink()
    # Dùng wget nếu có (resume tốt hơn); fallback urllib
    if shutil.which("wget"):
        _run(["wget", "--tries=3", "-c", "-O", str(RESNET101_PRETRAINED), PRETRAINED_URL])
    else:
        urllib.request.urlretrieve(PRETRAINED_URL, RESNET101_PRETRAINED)
    print(f"  [ok] {RESNET101_PRETRAINED.stat().st_size / 1e6:.1f} MB")


def verify() -> None:
    print("\n[verify] Smoke-test import CCNet đã patched ...")
    import sys
    if str(CCNET_ROOT) not in sys.path:
        sys.path.insert(0, str(CCNET_ROOT))
    from networks.ccnet import Seg_Model  # noqa: F401
    import torch
    m = Seg_Model(num_classes=4, criterion=None, pretrained_model=str(RESNET101_PRETRAINED), recurrence=2)
    x = torch.randn(1, 3, 64, 128)
    main_out, aux_out = m(x)
    print(f"  [ok] forward shapes: main={tuple(main_out.shape)} aux={tuple(aux_out.shape)}")
    print("  [ok] setup hoàn tất. Chạy train bằng:")
    print(f"       cd {PROJECT_ROOT}")
    print("       python -m scripts.new_branch.train --epochs 47 --batch-size 4")


def main() -> None:
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"CCNET_ROOT   = {CCNET_ROOT}")
    print(f"MODELS_DIR   = {MODELS_DIR}\n")
    clone_ccnet()
    patch_ccnet()
    download_pretrained()
    verify()


if __name__ == "__main__":
    main()
