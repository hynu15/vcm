# Hướng dẫn chi tiết tái hiện bài báo

## Semantic-Aware Video Compression for Automotive Cameras

> **Tác giả gốc**: Yiting Wang, Pak Hung Chan, Valentina Donzella
> **Nguồn**: IEEE Transactions on Intelligent Vehicles, Vol. 8, No. 6, June 2023
> **Mục đích tài liệu**: Hướng dẫn tái hiện chính xác 100% bài báo, kèm code thực thi được, sử dụng implementation gốc của CCNet.

---

## Mục lục

0. [Tóm tắt mục tiêu và phạm vi tái hiện](#0-tóm-tắt-mục-tiêu-và-phạm-vi-tái-hiện)
1. [Pipeline tổng quan của hệ thống](#1-pipeline-tổng-quan-của-hệ-thống)
2. [Cài đặt môi trường thực nghiệm](#2-cài-đặt-môi-trường-thực-nghiệm)
3. [Clone và tích hợp CCNet repo gốc](#3-clone-và-tích-hợp-ccnet-repo-gốc)
4. [Chuẩn bị dataset](#4-chuẩn-bị-dataset)
5. [Kiến trúc mạng Segmentation (Khối S1)](#5-kiến-trúc-mạng-segmentation-khối-s1)
6. [Huấn luyện mạng Segmentation](#6-huấn-luyện-mạng-segmentation)
7. [Tách luồng ROI / non-ROI (Khối S2)](#7-tách-luồng-roi--non-roi-khối-s2)
8. [Nén hai mức bằng FFmpeg (Khối S3)](#8-nén-hai-mức-bằng-ffmpeg-khối-s3)
9. [Bộ độ đo đánh giá](#9-bộ-độ-đo-đánh-giá)
10. [Hiện tượng artefact ở biên](#10-hiện-tượng-artefact-ở-biên)
11. [Tái hiện kết quả thực nghiệm](#11-tái-hiện-kết-quả-thực-nghiệm)
12. [Checklist tái hiện đầy đủ](#12-checklist-tái-hiện-đầy-đủ)
13. [Hướng mở rộng & gợi ý cải tiến](#13-hướng-mở-rộng--gợi-ý-cải-tiến)
14. [Tài liệu tham chiếu chính](#14-tài-liệu-tham-chiếu-chính)

---
* sử dụng conda activate sac  

## 0. Tóm tắt mục tiêu và phạm vi tái hiện

Bài báo đề xuất một khung nén video nhận thức ngữ nghĩa (**Semantic-Aware Compression - SAC**) cho camera ô tô. Ý tưởng cốt lõi: thay vì nén toàn bộ khung hình với cùng mức nén, hệ thống dùng segmentation để chia mỗi khung hình thành **vùng quan trọng (ROI)** – đường, xe, người đi bộ, vật thể động… – và **vùng ít quan trọng (non-ROI)** – trời, kiến trúc, cây cối, sau đó áp dụng hai mức CRF khác nhau khi nén bằng H.264/H.265. Mục đích là giữ chất lượng cao ở ROI – vùng quan trọng cho cảm nhận của xe tự lái – trong khi vẫn đạt được tỷ lệ nén tổng thể tương đương hoặc tốt hơn so với cách nén truyền thống.

Khi hoàn tất hướng dẫn này, bạn sẽ tái hiện được:

- Mạng segmentation 4 lớp (`ROI`, `sky`, `construction`, `nature`) dựa trên **ResNet-101 dilated + RCCA module** (Recurrent Criss-Cross Attention với `recurrence=2`) của CCNet.
- Pipeline tách 2 luồng (stream separation) với macroblock filter 16×16.
- Cơ chế nén hai mức (two-level compression) bằng FFmpeg `libx264` và `libx265` với hai giá trị CRF khác nhau cho ROI / non-ROI.
- Ba độ đo mới: **SA-PSNR**, **SA-SSIM** và **iIOU**.
- Đánh giá tác động của nén lên tác vụ segmentation hậu kỳ trên tập Cityscapes và KITTI-STEP.

**Các kết quả số chính cần tái hiện:**

| Chỉ số | Train | Val | Test |
|--------|-------|-----|------|
| mIOU (sau 47 epoch) | **87.97%** | **80.89%** | **79.08%** |

| Method | mIOU sau nén-giải nén | iIOU |
|--------|----------------------|------|
| SA-X264 (C_roi=18, C_non=27) | **90.56%** | **92.45%** |

---

## 1. Pipeline tổng quan của hệ thống

Khung làm việc SAC gồm 3 khối chính, được áp dụng frame-by-frame cho chuỗi video đầu vào `X = {x₁, x₂, …, x_t}`:

1. **Khối S1 – ROI Segmentation**: Mạng nơ-ron sinh mặt nạ phân vùng. Mạng chỉ học dự đoán non-ROI (sky/construction/nature); phần còn lại tự động trở thành ROI. Điều này giúp robust vì các đối tượng ROI rất đa dạng và liên tục thay đổi.
2. **Khối S2 – Stream Separation**: Từ mặt nạ phân vùng, sinh ra hai mặt nạ nhị phân `M_i` (ROI) và `M_n` (non-ROI). Trước khi áp vào ảnh, mặt nạ được làm tròn lên lưới 16×16 pixel (macroblock filter) để phù hợp với cấu trúc macroblock của H.264/H.265. **Quy tắc**: nếu một block 16×16 chứa ít nhất 1 pixel ROI thì cả block đó được coi là ROI. Hai luồng `S_i` và `S_n` được sinh ra bằng phép nhân Hadamard: `S_i = M_i ⊙ X`, `S_n = M_n ⊙ X`.
3. **Khối S3 – Two-level Compression**: Hai luồng `S_i` và `S_n` được nén song song bằng cùng codec (H.264 hoặc H.265) nhưng với hai giá trị CRF khác nhau. **CRF cho non-ROI > CRF cho ROI** (nén mạnh hơn cho vùng ít quan trọng). Sau giải nén, frame tái tạo là tổng của hai vùng: `x̂_t = îm_t + n̂m_t`.

Toàn bộ pipeline được thiết kế chạy song song, nên thời gian xử lý SAC chỉ tương đương nén truyền thống (đo được trên paper: ~0.184–0.258s/frame so với 0.211s/frame của X265 đơn luồng, trên Quadro P5000, chưa có tối ưu phần cứng).

```
┌─────────────────────┐    ┌─────────────────────────┐    ┌──────────────────────┐
│ S1: Segmentation    │    │ S2: Stream Separation   │    │ S3: Two-level        │
│ ResNet-101 + RCCA   │───▶│ Macroblock filter 16x16 │───▶│ Compression          │
│ → mask 4 lớp        │    │ → M_i, M_n → S_i, S_n   │    │ libx264 / libx265    │
└─────────────────────┘    └─────────────────────────┘    │ CRF_roi < CRF_non    │
                                                         └──────────────────────┘
```

---

## 2. Cài đặt môi trường thực nghiệm

### 2.1. Cấu hình phần cứng của bạn

| Mục đích | GPU | Khuyến nghị |
|----------|-----|-------------|
| Huấn luyện segmentation network (thuê) | **NVIDIA RTX 4090 (24 GB VRAM)** | Bài báo dùng Quadro P5000 (16 GB). RTX 4090 mạnh hơn ~3-4x, nên thời gian huấn luyện rút ngắn đáng kể (47 epoch chỉ còn vài tiếng). VRAM 24 GB cho phép tăng batch size lên 8 hoặc thậm chí 16 nếu muốn (giữ batch=4 để tái hiện chính xác paper). |
| Inference / test pipeline đơn giản | **NVIDIA GTX 1650 (4 GB VRAM)** | Đủ để chạy segmentation ở batch=1 và pipeline nén/giải nén. Lưu ý: 4 GB VRAM khá hạn chế với ResNet-101 ở full-res 1024×512 + RCCA. Có thể cần bật `torch.cuda.amp` (FP16) hoặc giảm input về 768×384 khi test. |

> **Lưu ý cấu hình GTX 1650**: Compute capability = 7.5 (Turing, sm_75). Khi cài PyTorch, chọn build có hỗ trợ sm_75. RTX 4090 có sm_89 (Ada Lovelace) – cần PyTorch 2.0+ hoặc PyTorch 1.13 + CUDA 11.8 trở lên.

### 2.2. Cấu hình phần mềm

- **OS**: Ubuntu 20.04 hoặc 22.04 (bài báo dùng Ubuntu 20).
- **Python**: 3.8 (Conda environment).
- **PyTorch**: 1.12 - 2.1 với CUDA 11.8 / 12.1 (tương thích cả RTX 4090 và GTX 1650).
- **FFmpeg**: build với x264 và x265.
- **Lưu trữ**: ≥100 GB (Cityscapes leftImg8bit_sequence ~324 GB nén lossless nhưng chỉ cần ~70 GB cho phần dùng trong thí nghiệm).

### 2.3. Các bước cài đặt được khuyến nghị

```bash
# 1) Tạo môi trường conda
conda create -n sac python=3.8 -y
conda activate sac

# 2) Cài PyTorch
# Cho RTX 4090 (server thuê) – CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Cho GTX 1650 (máy local) – CUDA 11.8 vẫn được:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3) Các thư viện phụ trợ
pip install numpy==1.23.5 pillow opencv-python scikit-image \
            tqdm tensorboard matplotlib scipy einops pyyaml ninja

# 4) Cài FFmpeg (cần x264 và x265)
sudo apt-get update
sudo apt-get install -y ffmpeg libx264-dev libx265-dev

# 5) Kiểm tra
ffmpeg -codecs 2>/dev/null | grep -E "264|265"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 3. Clone và tích hợp CCNet repo gốc

Để đảm bảo **chính xác 100%** với bài báo (vốn dùng kiến trúc CCNet tham chiếu [38]), ta sử dụng trực tiếp repo gốc của CCNet thay vì tự cài đặt lại.

### 3.1. Clone repo

```bash
mkdir -p ~/sac_project && cd ~/sac_project
git clone https://github.com/speedinghzl/CCNet.git
cd CCNet
```

### 3.2. Cấu trúc repo CCNet

```
CCNet/
├── cc_attention/
│   ├── __init__.py
│   └── functions.py           # CrissCrossAttention module (pure PyTorch)
├── dataset/                   # Dataloaders cho Cityscapes
├── loss/
│   ├── criterion.py           # CriterionDSN
│   ├── loss.py                # OhemCrossEntropy2d
│   └── lovasz_losses.py
├── networks/
│   ├── ccnet.py               # ResNet101 + RCCAModule (Cityscapes-trained)
│   ├── deeplabv3.py
│   └── pspnet.py
├── utils/
│   ├── pyt_utils.py
│   ├── encoding.py
│   └── ...
├── engine.py
├── evaluate.py
├── train.py                   # Script train gốc
├── test.py
└── run_local.sh
```

### 3.3. Điểm cốt lõi của implementation gốc

#### a) `cc_attention/functions.py` – Pure PyTorch CCA (không cần CUDA build)

Đây là phiên bản pure-PyTorch (borrowed từ Serge-weihao/CCNet-Pure-Pytorch) – chạy được trên cả RTX 4090 và GTX 1650 mà không cần biên dịch CUDA extension:

```python
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0)\
        .unsqueeze(0).repeat(B*W, 1, 1)

class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim,      kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width, -1, height).permute(0,2,1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width).permute(0,2,1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width, -1, height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width, -1, height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width))\
                       .view(m_batchsize, width, height, height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,  height, height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0,2,1)).view(m_batchsize, width,  -1, height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0,2,1)).view(m_batchsize, height, -1, width).permute(0,2,1,3)
        return self.gamma * (out_H + out_W) + x
```

#### b) `networks/ccnet.py` – RCCA + ResNet101

**Chi tiết quan trọng** mà tôi đã đọc từ source code gốc:

```python
class RCCAModule(nn.Module):
    """Recurrent Criss-Cross Attention Module"""
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        inter_channels = in_channels // 4   # 2048 // 4 = 512
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            InPlaceABNSync(inter_channels))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, 3, padding=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, bias=True))

    def forward(self, x, recurrence=2):       # <<< QUAN TRỌNG: paper dùng R=2
        output = self.conva(x)
        for i in range(recurrence):           # cùng MỘT module CCA, chạy 2 lần
            output = self.cca(output)         # → shared parameters
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output
```

> **Chú ý lớn**: 2 lần CCA trong CCNet là **Recurrent CCA (shared parameters)**, KHÔNG phải 2 module CCA riêng biệt. Điều này giảm số lượng tham số mạng mà vẫn cho mỗi pixel "thấy" toàn bộ ảnh sau 2 vòng (do bản chất criss-cross). Mọi tài liệu/blog mô tả "stack 2 CCA modules" với 2 sets of params là **sai** so với implementation gốc.

#### c) ResNet-101 đã sửa đổi (deep stem + dilated)

CCNet không dùng ResNet-101 chuẩn của torchvision; thay vào đó:

- **Deep stem**: 3 conv 3×3 (3→64→64→128) thay cho 1 conv 7×7 đầu tiên.
- **layer1**: 3 bottleneck blocks, output 256 channels, stride 1.
- **layer2**: 4 bottleneck blocks, output 512, stride 2.
- **layer3**: 23 bottleneck blocks, output 1024, **stride=1, dilation=2** (không downsample).
- **layer4**: 3 bottleneck blocks, output 2048, **stride=1, dilation=4, multi_grid=(1,1,1)**.
- **Output stride tổng**: 8 (input 1024×512 → feature map 128×64).

#### d) Auxiliary head (DSN – Deep Supervision)

Code gốc thêm một head phụ từ `layer3` (1024 channels) để cải thiện hội tụ:

```python
self.dsn = nn.Sequential(
    nn.Conv2d(1024, 512, 3, padding=1),
    InPlaceABNSync(512),
    nn.Dropout2d(0.1),
    nn.Conv2d(512, num_classes, kernel_size=1))
```

Loss tổng = `loss_main + 0.4 * loss_aux`.

### 3.4. Thay InPlaceABN bằng BatchNorm chuẩn (KHUYẾN NGHỊ)

Code gốc dùng `inplace_abn` (cần build C++/CUDA extension, hay vỡ tương thích). Để đơn giản và đảm bảo chạy được trên cả RTX 4090 và GTX 1650, **thay tất cả `InPlaceABNSync` bằng `nn.BatchNorm2d + nn.ReLU`**:

```python
# Trong networks/ccnet.py – sửa đầu file
import torch.nn as nn
# Bỏ: from inplace_abn import InPlaceABN, InPlaceABNSync
# Bỏ: BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')

# Thay bằng:
class BNReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.bn(x))

# Sau đó thay mọi `InPlaceABNSync(N)` thành `BNReLU(N)` ;
# và mọi `BatchNorm2d(N)` (functools.partial) thành `nn.BatchNorm2d(N)`.
```

> Lý do được giữ kết quả: InPlaceABN chỉ là tối ưu memory/speed, KHÔNG ảnh hưởng đến accuracy. Paper báo cáo các con số mIOU vẫn đạt được với BN chuẩn.

---

## 4. Chuẩn bị dataset

### 4.1. Cityscapes (dataset chính)

**Lý do chọn**: Cityscapes là dataset duy nhất ở thời điểm đó cung cấp **temporal sequences** (đoạn video liên tục) kèm segmentation masks – điều kiện bắt buộc để đánh giá nén video dựa trên inter-frame prediction.

- Đăng ký tại [https://www.cityscapes-dataset.com](https://www.cityscapes-dataset.com) và tải các gói:
  - `leftImg8bit_sequence_trainvaltest.zip` – chứa các chuỗi 30 frame liên tục, độ phân giải 2048×1024 RGB (đây là input cho pipeline nén).
  - `leftImg8bit_trainvaltest.zip` – frame chú thích (frame thứ 20 trong mỗi đoạn).
  - `gtFine_trainvaltest.zip` – nhãn segmentation tinh.

**Cấu trúc thư mục đề xuất:**

```
~/sac_project/datasets/cityscapes/
├── leftImg8bit/             # frame chú thích (1 frame / sequence)
│   ├── train/<city>/*.png
│   ├── val/<city>/*.png
│   └── test/<city>/*.png
├── leftImg8bit_sequence/    # 30 frame / sequence để test nén video
│   └── ...
└── gtFine/                  # nhãn segmentation
    ├── train/<city>/*_gtFine_labelIds.png
    ├── val/...
    └── test/...
```

### 4.2. Ánh xạ nhãn 4 lớp (ROI / sky / construction / nature)

Bài báo gom 19 lớp gốc của Cityscapes thành 4 lớp đầu ra cho mạng segmentation. **ROI = mọi thứ không phải 3 lớp non-ROI sau đây:**

| Lớp đầu ra | Lớp gốc Cityscapes (trainId / name) | Vai trò |
|------------|--------------------------------------|---------|
| `sky` | sky (10) | non-ROI |
| `construction` | building (2), wall (3), fence (4), guard rail, bridge, tunnel | non-ROI |
| `nature` | vegetation (8), terrain (9) | non-ROI |
| `ROI` | road (0), sidewalk (1), pole, traffic light, traffic sign, person (11), rider (12), car (13), truck (14), bus (15), train (16), motorcycle (17), bicycle (18), object, ground, parking, dynamic, unlabeled, … | ROI (được nén nhẹ hơn) |

Cụ thể, bài báo nêu rõ: *"non-ROI là kết hợp 3 lớp construction (building, wall, fence, guard rail, bridge, tunnel), nature (vegetation, terrain), sky; ROI là phần còn lại"*.

**Script ánh xạ nhãn:**

```python
# datasets/label_mapping.py
import numpy as np

# Cityscapes labelId → 4-class id  (0=ROI, 1=sky, 2=construction, 3=nature)
LABEL_MAP = np.full(256, 0, dtype=np.uint8)   # mặc định = ROI

# sky
LABEL_MAP[23] = 1
# construction (building=11, wall=12, fence=13, guard rail=14, bridge=15, tunnel=16)
for k in [11, 12, 13, 14, 15, 16]: LABEL_MAP[k] = 2
# nature (vegetation=21, terrain=22)
for k in [21, 22]: LABEL_MAP[k] = 3

def to_4class(label_img):
    """label_img: numpy uint8 (H,W) chứa Cityscapes labelIds (0..33)."""
    return LABEL_MAP[label_img]
```

> **Lưu ý**: Mạng segmentation thực tế chỉ học predict 3 lớp non-ROI. Khi inference, mọi pixel không thuộc 3 lớp này được gán là ROI. Đây là quyết định thiết kế cố ý để tăng độ robust vì các đối tượng ROI rất đa dạng (xe, người, biển báo, vật động…).

### 4.3. KITTI-STEP (dataset validate)

KITTI-STEP (Weber et al., NeurIPS 2021) cung cấp các đoạn video lái xe nông thôn/đô thị với mask segmentation cho từng frame liên tiếp. Dùng để xác nhận tính tổng quát hóa của phương pháp.

- Tải tại [https://www.cvlibs.net/datasets/kitti/eval_step.php](https://www.cvlibs.net/datasets/kitti/eval_step.php) hoặc qua DeepLab2.
- Áp dụng cùng ánh xạ 4 lớp như Cityscapes.

---

## 5. Kiến trúc mạng Segmentation (Khối S1)

Mạng được mô tả ở Hình 3 của bài báo. Cấu trúc tổng thể bao gồm 5 bước, được tổng hợp ngắn gọn bằng phương trình:

```
S = f ( R(X) + H'' )
```

Trong đó `R` là feature extractor (ResNet-101 dilated), `H''` là feature dày đặc đầu ra của RCCA module (CCA chạy 2 lần với shared params), và `f` là phân lớp segmentation cuối cùng.

### 5.1. Tiền xử lý input

1. **Đầu vào**: ảnh RGB kích thước 2048×1024 (chuẩn Cityscapes).
2. **Downsize** xuống 1024×512 (đầu ra segmentation cũng ở kích thước này theo Hình 3).
3. **Chuyển grayscale** (theo Section III-B: *"the sensor generated video sequences X are converted to lower resolution images and transformed to greyscale through a deep neural network"*). Cách tiêu chuẩn: dùng phép biến đổi luma `Y = 0.299R + 0.587G + 0.114B` rồi nhân lên 3 kênh để giữ tương thích đầu vào ResNet.
4. **Chuẩn hóa**: trừ ImageNet mean `(0.485, 0.456, 0.406)`, chia std `(0.229, 0.224, 0.225)`.

### 5.2. Backbone: ResNet-101 đã sửa đổi cho segmentation

Đã mô tả ở [§3.3 c](#c-resnet-101-đã-sửa-đổi-deep-stem--dilated). Lấy nguyên từ `networks/ccnet.py` của CCNet repo:

- Bỏ tầng pooling cuối và FC.
- Trong layer3 và layer4, stride=1 + dilated conv (rates 2 và 4 tương ứng). **Output stride = 8.**
- Với input 1024×512 → feature map `X̄` kích thước 128×64 với 2048 kênh.

**Khởi tạo**: Tải pretrained ImageNet weights. CCNet cung cấp link tải resnet101-imagenet.pth tại README. Nếu không tải được, có thể init từ `torchvision.models.resnet101(pretrained=True)` và map các trọng số sang stem mới (chỉ dùng được conv1 nếu giữ stem cũ; với deep stem thì init random và chấp nhận hội tụ chậm hơn ~1-2 epoch).

### 5.3. RCCA Module – linh hồn của mạng

Đã trình bày code đầy đủ ở [§3.3 b](#b-networksccnetpy--rcca--resnet101). Tóm tắt:

```
X̄ (2048,128,64) → conva (Conv3x3 → 512) → CCA → CCA → convb (Conv3x3 → 512)
                          (← cùng module CCA, shared params, R=2 lần)
                                       │
                                       ▼
                          concat([X̄, output], dim=1)  → (2048+512, 128, 64) = 2560 ch
                                       │
                                       ▼
                          Conv3x3 → 512 → BN → ReLU → Dropout(0.1) → Conv1x1 → 4 classes
                                       │
                                       ▼ bilinear upsample x8
                                segmentation map (4, 1024, 512)
```

**Tổng số tham số ước tính**: ~70M (ResNet-101 ~44M + RCCA module ~25M + aux head ~1M).

### 5.4. Wrapper hoàn chỉnh (tích hợp với CCNet repo)

Tạo file `sac_segnet.py` để wrap mạng CCNet với:
- Số lớp = 4
- Recurrence = 2
- Input grayscale-as-3channel
- Aux head DSN cho training

```python
# sac_segnet.py
import sys
sys.path.insert(0, '/path/to/CCNet')   # đường dẫn tới repo CCNet đã clone

from networks.ccnet import Seg_Model    # tên class trong networks/ccnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_sac_segnet(num_classes=4, recurrence=2, criterion=None, pretrained='resnet101-imagenet.pth'):
    """
    Trả về model SAC segmentation = CCNet với 4 lớp đầu ra.
    """
    model = Seg_Model(
        num_classes=num_classes,
        criterion=criterion,
        pretrained_model=pretrained,
        recurrence=recurrence)
    return model

def rgb_to_gray_3ch(x):
    """x: (B,3,H,W) RGB normalized → grayscale lặp 3 lần (vẫn 3 kênh)."""
    # Luma: Y = 0.299 R + 0.587 G + 0.114 B
    y = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
    return y.repeat(1, 3, 1, 1)
```

---

## 6. Huấn luyện mạng Segmentation

### 6.1. Hàm loss BCE-Dice

Bài báo dùng tổ hợp Binary Cross-Entropy (BCE) và Dice loss để xử lý mất cân bằng lớp và tăng độ nhạy với vùng nhỏ.

BCE loss với phân phối lớp thật `s` và dự đoán `ŝ`:

```
L_BCE(s, ŝ) = − ( s·log(ŝ) + (1−s)·log(1−ŝ) )
```

Dice loss giữa nhãn `s` và dự đoán `p̂`, với chỉ số làm trơn = 1 để tránh chia 0:

```
L_Dice(s, p̂) = 1 − ( 2·|s ∩ p̂| + 1 ) / ( |s ∪ p̂| + 1 )
```

Tổng hợp:

```
L_BCE-Dice = L_BCE(s, ŝ) + L_Dice(s, p̂)
```

**Implementation (multi-class biến BCE thành Cross-Entropy thông thường):**

```python
# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (B,C,H,W); target: (B,H,W) với class id 0..C-1
        ce_loss = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index).unsqueeze(1).float()
        target_clamped = target.clone(); target_clamped[target == self.ignore_index] = 0
        target_1h = F.one_hot(target_clamped, self.num_classes).permute(0,3,1,2).float() * valid
        inter = (probs * target_1h).sum(dim=(2,3))
        union = (probs * valid).sum(dim=(2,3)) + target_1h.sum(dim=(2,3))
        dice = 1 - (2*inter + self.smooth) / (union + self.smooth)
        return ce_loss + dice.mean()

class BCEDiceWithAux(nn.Module):
    """Khớp với CCNet: main loss + 0.4 * aux loss."""
    def __init__(self, **kwargs):
        super().__init__()
        self.main = BCEDiceLoss(**kwargs)
        self.aux  = BCEDiceLoss(**kwargs)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        main_pred = F.interpolate(preds[0], size=(h,w), mode='bilinear', align_corners=True)
        aux_pred  = F.interpolate(preds[1], size=(h,w), mode='bilinear', align_corners=True)
        return self.main(main_pred, target) + 0.4 * self.aux(aux_pred, target)
```

### 6.2. Tham số huấn luyện (nguyên văn từ bài báo)

| Tham số | Giá trị |
|---------|---------|
| Optimizer | **Adam** (Adaptive Moment Estimation) |
| Learning rate | **3 × 10⁻⁴** |
| Betas (β1, β2) | **(0.9, 0.999)** |
| Weight decay | **1 × 10⁻⁵** |
| Số epoch | **47** |
| Số iteration / epoch | **2974** |
| Batch size | **4** |
| Số lớp đầu ra | 4 (ROI, sky, construction, nature) |
| Input size | 1024 × 512 grayscale (downsized từ 2048×1024 RGB) |

> **Lưu ý**: 2974 iter × 4 batch = 11896 ảnh/epoch, gần đúng với 11900 ảnh huấn luyện của tập train Cityscapes `leftImg8bit_sequence` (sau khi chọn 1 frame chú thích cho mỗi đoạn).

> **Với RTX 4090**: Batch=4 chạy nhanh, 47 epoch ước tính ~3-5 giờ. Có thể bật `torch.cuda.amp.autocast` để tăng tốc 1.5-2x mà không ảnh hưởng kết quả.

### 6.3. Vòng huấn luyện hoàn chỉnh

```python
# train_sac.py
import torch, os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from sac_segnet import build_sac_segnet, rgb_to_gray_3ch
from losses import BCEDiceWithAux
from dataset.cityscapes_sac import CityscapesSAC   # bạn tự viết, xem §6.4

def main():
    device = "cuda"
    criterion = BCEDiceWithAux(num_classes=4)
    model = build_sac_segnet(num_classes=4, recurrence=2, criterion=criterion).to(device)

    optimizer = Adam(model.parameters(), lr=3e-4,
                     betas=(0.9, 0.999), weight_decay=1e-5)

    train_ds = CityscapesSAC(
        root="datasets/cityscapes", split="train",
        size=(1024, 512), to_grayscale=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=8, drop_last=True, pin_memory=True)

    scaler = GradScaler()
    os.makedirs("ckpts", exist_ok=True)

    for epoch in range(47):
        model.train()
        for it, (img, mask4) in enumerate(train_loader):
            img, mask4 = img.to(device, non_blocking=True), mask4.to(device, non_blocking=True).long()
            img_gray = rgb_to_gray_3ch(img)

            optimizer.zero_grad()
            with autocast():
                preds = model(img_gray, mask4)     # CCNet model trả về (main, aux) hoặc loss tùy build
                # Nếu model tự tính loss qua self.criterion, lấy thẳng loss; nếu không:
                # loss = criterion(preds, mask4)
                loss = preds if isinstance(preds, torch.Tensor) else criterion(preds, mask4)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if it % 100 == 0:
                print(f"epoch {epoch} iter {it}/{len(train_loader)} loss {loss.item():.4f}")

        torch.save(model.state_dict(), f"ckpts/ckpt_epoch{epoch:02d}.pth")

if __name__ == "__main__":
    main()
```

### 6.4. Dataset class mẫu

```python
# dataset/cityscapes_sac.py
import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from datasets.label_mapping import to_4class

class CityscapesSAC(Dataset):
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

    def __init__(self, root, split='train', size=(1024, 512), to_grayscale=True):
        self.imgs = sorted(glob.glob(os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")))
        self.size = size  # (W, H)
        self.to_grayscale = to_grayscale
        self.tf_img = T.Compose([T.ToTensor(), T.Normalize(self.MEAN, self.STD)])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img_path = self.imgs[i]
        lbl_path = img_path.replace("leftImg8bit", "gtFine").replace("_gtFine.png", "_gtFine_labelIds.png")
        # Patch the path mapping (Cityscapes naming):
        lbl_path = img_path.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit.png", "_gtFine_labelIds.png")

        img = Image.open(img_path).convert("RGB").resize(self.size, Image.BILINEAR)
        lbl = Image.open(lbl_path).resize(self.size, Image.NEAREST)
        lbl_np = to_4class(np.array(lbl, dtype=np.uint8))   # → 0..3

        img_t = self.tf_img(img)
        return img_t, torch.from_numpy(lbl_np)
```

### 6.5. Kết quả mong đợi sau huấn luyện

Bài báo báo cáo các giá trị mIOU sau 47 epoch:

| Phân chia | mIOU |
|-----------|------|
| Train | **87.97%** |
| Validation | **80.89%** |
| Test | **79.08%** |

Nếu sai số ≤ 1.5 điểm là khớp tốt; > 3 điểm cần kiểm tra: (a) pretrained weights, (b) input size, (c) DSN aux loss, (d) random seed.

---

## 7. Tách luồng ROI / non-ROI (Khối S2)

Sau khi có mask 4 lớp `S`, thực hiện theo 3 bước (Hình 4 của bài báo):

1. **Binary masking**: gộp sky + construction + nature → non-ROI; phần còn lại → ROI. Tạo ảnh nhị phân kích thước 1024×512, sau đó upsample về 2048×1024 bằng nearest-neighbor để khớp kích thước frame gốc.
2. **Macroblock filter (16×16)**: chia frame thành lưới các block 16×16. Với mỗi block, **nếu có ít nhất một pixel ROI thì toàn bộ block được gán nhãn ROI**. Quy tắc này quan trọng vì nó tránh làm mất pixel ROI ở biên (theo bài báo: *"the loss of any ROI pixels belonging to critical objects in the frames is reduced"*).
3. **Sinh hai mặt nạ nhị phân** `M_i` (ROI) và `M_n = 1 − M_i` (non-ROI). Áp dụng vào frame gốc bằng nhân Hadamard:

```
S_i = M_i ⊙ X       ;       S_n = M_n ⊙ X
```

Pixel ngoài mặt nạ của mỗi luồng được đặt = 0 (đen).


## 8. Nén hai mức bằng FFmpeg (Khối S3)

### 8.1. Nguyên tắc

Mỗi luồng (`S_i` và `S_n`) được lưu thành chuỗi PNG hoặc YUV, sau đó nén **độc lập** bằng FFmpeg với codec `libx264` hoặc `libx265`, dùng chế độ Constant Rate Factor (CRF) – chế độ biến tỉ lệ bit để giữ chất lượng cố định. CRF càng cao, nén càng mạnh, chất lượng càng thấp. **Yêu cầu cốt lõi**: `CRF_non > CRF_roi`.

### 8.2. Bảng giá trị CRF được dùng trong bài báo

| Phương pháp | CRF cho ROI | CRF cho non-ROI | Tỷ lệ nén xấp xỉ |
|-------------|-------------|-----------------|------------------|
| H.264 (truyền thống) | 23 (a) | 23 (a) | ≈ 1:250 |
| H.265 (truyền thống) | 28 (b) | 28 (b) | ≈ 1:375 |
| **SA-X264** | **18 (c)** | **27 (d)** | ≈ 1:250 (khớp H.264) |
| **SA-X265** | **23 (e)** | **32 (f)** | ≈ 1:375 (khớp H.265) |

Mọi so sánh đều phải đảm bảo **CÙNG tỷ lệ nén tổng** để công bằng. Khi bạn muốn nén mạnh hơn cho non-ROI (giảm thêm dung lượng ≈6% như paper với SA-X265), chỉ cần tăng `CRF_non`.

### 8.3. Lệnh FFmpeg minh họa

Giả sử bạn đã tách frame và lưu hai chuỗi ảnh tại `stream_roi/` và `stream_non/` với pattern `frame_%05d.png`:

```bash
# === SA-X264: nén luồng ROI với CRF=18 (chất lượng cao) ===
ffmpeg -y -framerate 30 -i stream_roi/frame_%05d.png \
       -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
       -x264-params "keyint=30:min-keyint=30" \
       out_roi.mp4

# === SA-X264: nén luồng non-ROI với CRF=27 (nén mạnh) ===
ffmpeg -y -framerate 30 -i stream_non/frame_%05d.png \
       -c:v libx264 -crf 27 -preset medium -pix_fmt yuv420p \
       -x264-params "keyint=30:min-keyint=30" \
       out_non.mp4

# === SA-X265: CRF 23 cho ROI ===
ffmpeg -y -framerate 30 -i stream_roi/frame_%05d.png \
       -c:v libx265 -crf 23 -preset medium -pix_fmt yuv420p \
       -x265-params "keyint=30:min-keyint=30" \
       out_roi_h265.mp4

# === SA-X265: CRF 32 cho non-ROI ===
ffmpeg -y -framerate 30 -i stream_non/frame_%05d.png \
       -c:v libx265 -crf 32 -preset medium -pix_fmt yuv420p \
       -x265-params "keyint=30:min-keyint=30" \
       out_non_h265.mp4
```

> Bài báo không nêu rõ `-preset`, `-gop`. Đặt `keyint=30` (GOP 30) khớp framerate 30fps Cityscapes là an toàn. Dùng cùng `-preset medium` ở mọi so sánh để công bằng.

### 8.4. Giải nén và ghép lại frame

Sau khi giải nén, bạn có hai chuỗi frame `îm_t` và `n̂m_t`. Ghép chúng lại để có frame tái tạo cuối cùng:

```
x̂_t = îm_t + n̂m_t
```

Do mỗi luồng có pixel ngoài mask = 0, phép cộng đơn giản (saturating add) sẽ phục hồi toàn bộ frame. Tuy nhiên codec không tái tạo đúng 0 tuyệt đối, nên trong thực hành **nên ghép bằng mask gốc**:

```python
import numpy as np
def recombine(roi_frame, non_frame, roi_mask):
    """roi_mask đã saved cùng metadata để decoder dùng lại."""
    out = np.where(roi_mask[..., None] == 1, roi_frame, non_frame)
    return out.astype(np.uint8)

'''
> **Lưu ý quan trọng**: Một thực tế đáng chú ý là bài báo nói "phép cộng" (Eq. 7), điều này hoạt động vì luồng nào không có dữ liệu thì pixel = 0. Để tránh sai số do codec, nên truyền kèm mask để dùng `np.where` khi ghép. Mask cần được nén lossless (PNG hoặc run-length) hoặc tái tạo lại bằng segmentation ở phía decoder nếu băng thông yêu cầu thấp.


## 9. Bộ độ đo đánh giá

### 9.1. Độ đo truyền thống

- **PSNR** (Peak Signal-to-Noise Ratio): càng cao càng tốt, đơn vị dB.
- **SSIM** (Structural Similarity Index): trong [0, 1], càng gần 1 càng tốt.
- **Compression ratio**: tỉ lệ kích thước file gốc / file nén.

### 9.2. Chỉ số trọng số nén (compression ratio indexes)

Cho `C_roi`, `C_non` lần lượt là CRF của ROI và non-ROI; `(S_i, S_n)` là SSIM tính riêng cho ROI và non-ROI; `(P_i, P_n)` là PSNR tương tự.

```
r_roi = C_roi / (C_non + C_roi)
r_non = C_non / (C_non + C_roi)
```

Vì `C_roi < C_non`, ta luôn có `r_non > r_roi`. Hai trọng số này được dùng để cân chỉnh độ quan trọng tương đối khi tổng hợp chất lượng.

### 9.3. SA-SSIM và SA-PSNR (mới đề xuất)

```
SA-SSIM = r_non · S_i + r_roi · S_n
SA-PSNR = r_non · P_i + r_roi · P_n
```

> **Cách đánh trọng số quan trọng**: Trọng số LỚN (`r_non`) được gắn với chất lượng của ROI (`S_i`, `P_i`), trọng số NHỎ (`r_roi`) gắn với non-ROI. Điều này có nghĩa: chất lượng ROI được "tăng cân" trong tổng hợp – đúng triết lý của bài báo: ROI quan trọng hơn.

### 9.4. iIOU (region-of-interest IoU)

Cho `GT_roi` là ground-truth pixel ROI và `Pre_roi` là dự đoán segmentation cho ROI:

```
IoU = |GT ∩ Pre| / |GT ∪ Pre|
mIoU = ( Σ_{k=1}^{m} IoU_k ) / m
iIoU = |GT_roi ∩ Pre_roi| / |GT_roi ∪ Pre_roi|
```

iIOU đo riêng độ chính xác segmentation đối với vùng ROI – phần quan trọng cho lái xe – nên là chỉ số tốt nhất để đánh giá hiệu ứng cuối cùng của nén lên perception.

```

### 11.1. Bảng II – Chất lượng nén trên Cityscapes

So sánh nén truyền thống vs SAC với cùng tỉ lệ nén. Bạn cần xác nhận tái hiện được xu hướng sau:

| Phương pháp | C_roi | C_non | SA-PSNR (↑) | SA-SSIM (↑) |
|-------------|-------|-------|-------------|-------------|
| H.264 | 23 | 23 | Tham chiếu | Tham chiếu |
| **SA-X264** | **18** | **27** | **+ 2.864 dB** so với H.264 | **+ 0.008** so với H.264 |
| H.265 | 28 | 28 | Tham chiếu | Tham chiếu |
| **SA-X265** | **23** | **32** | Cao hơn H.265 | Tương đương / cao hơn H.265 |

### 11.2. Bảng IV – Segmentation hậu nén
Dùng đúng mạng segmentation đã huấn luyện (Mục 6), apply lên các frame đã nén-giải nén:

| Method | C_roi | C_non | SA-PSNR (dB) | SA-SSIM | mIOU (%) | iIOU (%) |
|--------|-------|-------|--------------|---------|----------|----------|
| H.264 | 23 | 23 | 45.18 | 0.98 | 87.86 | 92.00 |
| **SA-X264** | **18** | **27** | **48.05** | **0.99** | **90.56** | **92.45** |
| H.265 | 28 | 28 | 43.80 | 0.98 | 85.71 | 91.43 |
| SA-X265 | 23 | 27 | 47.01 | 0.99 | 87.48 | 92.04 |

**Các kết quả nổi bật cần khớp:**

- **SA-X264 đạt mIOU = 90.56% và iIOU = 92.45% – cao nhất** trong 4 phương pháp.
- SA-X264 cải thiện **2.7 điểm mIOU** so với H.264 truyền thống và **0.45 điểm iIOU**.
- SA-X264 có SA-PSNR **+ 2.864 dB** và SA-SSIM **+ 0.008** so với H.264 với cùng tỉ lệ nén.

## 12. Checklist tái hiện đầy đủ

Đánh dấu lần lượt từng mục trong quá trình thực hiện:

- [ ] Tải Cityscapes (`leftImg8bit_sequence` + `leftImg8bit` + `gtFine`) và KITTI-STEP.
- [ ] Viết script ánh xạ 19 lớp gốc → 4 lớp `{ROI, sky, construction, nature}`.
- [ ] Cài Conda + PyTorch 2.0+ + FFmpeg (chạy được trên cả RTX 4090 và GTX 1650).
- [ ] **Clone CCNet repo gốc** (`speedinghzl/CCNet`) và thay `InPlaceABNSync` → `BNReLU`.
- [ ] Cài đặt `SACSegNet = CCNet(num_classes=4, recurrence=2)` dùng `RCCAModule` với CCA shared params.
- [ ] (Trên RTX 4090) Huấn luyện 47 epoch với Adam(lr=3e-4, β=(0.9, 0.999), wd=1e-5), batch 4, BCE-Dice loss + 0.4 × aux loss.
- [ ] Đạt mIOU ≈ **87.97% train / 80.89% val / 79.08% test**.
- [ ] (Trên GTX 1650 cũng test được) Cài đặt stream separation: binary mask → upsample 2048×1024 → macroblock filter 16×16 → `M_i`, `M_n` → `S_i`, `S_n`.
- [ ] Nén song song hai luồng bằng FFmpeg với cặp CRF: **(18, 27)** cho SA-X264, **(23, 32)** cho SA-X265.
- [ ] Giải nén và ghép frame bằng phép cộng / `np.where` với mask đi kèm.
- [ ] Cài đặt 3 độ đo mới: **SA-PSNR**, **SA-SSIM**, **iIOU**.
- [ ] Đánh giá hai cách: (a) chất lượng nén (PSNR/SSIM/SA-PSNR/SA-SSIM), (b) tác động đến segmentation (mIOU/iIOU) – tái nạp model segmentation lên frame tái tạo.
- [ ] Lập bảng kết quả khớp Bảng II, III, IV của bài báo (sai số ≤ 1 điểm là chấp nhận được).
- [ ] Quan sát artefact ở biên (đôi khi xuất hiện) và xác nhận chúng không làm hỏng segmentation.

---
*— Hết hướng dẫn —*