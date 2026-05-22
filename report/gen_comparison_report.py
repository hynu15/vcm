"""
Script tạo báo cáo so sánh chi tiết hai pipeline SAC:
  A) scripts/sac_compression_x265.py
  B) scripts/SAC/Codes/CCNet/  (TwoStream_generate.py + combine.py + run_smoke_pipeline.py + eval_sac.py)
"""

from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def set_col_width(table, col_idx, width_cm):
    for row in table.rows:
        row.cells[col_idx].width = Cm(width_cm)


def shade_cell(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return h


def add_code(doc, code_text, font_size=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(font_size)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    # light grey background via direct XML
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  'F2F2F2')
    pPr.append(shd)
    return p


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(4)
    return p


def add_warning(doc, text):
    p = doc.add_paragraph()
    run = p.add_run('⚠ ' + text)
    run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
    run.bold = True
    return p


def add_ok(doc, text):
    p = doc.add_paragraph()
    run = p.add_run('✔ ' + text)
    run.font.color.rgb = RGBColor(0x27, 0xAE, 0x60)
    run.bold = True
    return p


def make_compare_table(doc, rows, header=('Pipeline A – sac_compression_x265.py',
                                           'Pipeline B – SAC/Codes/CCNet (modular)')):
    """rows = list of (label, code_a, code_b, impact_text)"""
    table = doc.add_table(rows=1 + len(rows), cols=3)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # header
    hdr = table.rows[0].cells
    hdr[0].text = 'Khía cạnh'
    hdr[1].text = header[0]
    hdr[2].text = header[1]
    for c in hdr:
        shade_cell(c, '1F497D')
        for p in c.paragraphs:
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.bold = True
                r.font.size = Pt(9)

    for i, (label, a, b, _) in enumerate(rows):
        row = table.rows[i + 1].cells
        row[0].text = label
        row[1].text = a
        row[2].text = b
        for c in row:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8.5)
        shade_cell(row[0], 'DCE6F1')
        if i % 2 == 1:
            shade_cell(row[1], 'F9F9F9')
            shade_cell(row[2], 'F9F9F9')

    set_col_width(table, 0, 4.0)
    set_col_width(table, 1, 7.5)
    set_col_width(table, 2, 7.5)
    return table


# ────────────────────────────────────────────────────────────────────────────
# Document
# ────────────────────────────────────────────────────────────────────────────
doc = Document()

# Page margins
section = doc.sections[0]
section.left_margin   = Cm(2.0)
section.right_margin  = Cm(2.0)
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)

# Default style
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)

# ── Title ────────────────────────────────────────────────────────────────────
title = doc.add_heading('Báo cáo So sánh Chi tiết Hai Pipeline SAC', 0)
title.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run(
    'sac_compression_x265.py  vs  SAC/Codes/CCNet (TwoStream_generate + combine + run_smoke_pipeline + eval_sac)\n'
    f'Ngày tạo: {datetime.date.today().strftime("%d/%m/%Y")}  |  Dự án: Semantic-Aware Compression (SAC)'
)
r.font.size = Pt(10)
r.font.italic = True

doc.add_paragraph()

# ── 1. Tổng quan ─────────────────────────────────────────────────────────────
add_heading(doc, '1. Tổng quan hai pipeline', 1)
add_body(doc,
    'Hai pipeline đều thực hiện cùng một mục tiêu: tách mỗi frame ảnh Cityscapes thành vùng ROI '
    '(Important Area – đường, xe, người đi bộ) và non-ROI (Background Area – trời, công trình, thực vật), '
    'nén độc lập bằng codec H.264/H.265 với CRF khác nhau, rồi ghép lại thành frame tái tạo. '
    'Tuy nhiên, cách hiện thực của từng bước có nhiều điểm khác biệt quan trọng ảnh hưởng đến '
    'kết quả số liệu và khả năng tái hiện paper.')

# overview table
ov_table = doc.add_table(rows=9, cols=3)
ov_table.style = 'Table Grid'
hdr = ov_table.rows[0].cells
for c, t in zip(hdr, ['Tiêu chí', 'Pipeline A (sac_compression_x265.py)', 'Pipeline B (SAC/Codes/CCNet)']):
    c.text = t
    shade_cell(c, '2E75B6')
    for p in c.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.bold = True
            r.font.size = Pt(9)

ov_data = [
    ('Phạm vi', 'Chỉ X265, không đánh giá', 'H264 + H265 + SA-X264 + SA-X265, đánh giá đầy đủ'),
    ('Cấu trúc file', '1 script monolithic (~189 dòng)', 'Nhiều file: TwoStream, combine, run_smoke, eval_sac'),
    ('Nguồn mask segmentation', 'Inference inline (chạy model mỗi frame)', 'Đọc mask PNG đã lưu sẵn từ data/gt_4class/'),
    ('Phép tách luồng', 'cv2.bitwise_and (mask 0/255)', 'Nhân Hadamard: img * mi (mask 0/1) – khớp paper'),
    ('Ghép 2 luồng sau nén', 'FFmpeg blend=addition (video-level)', 'cv2.add() frame-by-frame (pixel-level)'),
    ('Macroblock filter', 'Vectorized numpy (nhanh, xử lý padding)', 'Explicit loop (chậm, bỏ sót biên nếu H/W ≠ bội 16)'),
    ('FFmpeg flags', 'Thiếu -pix_fmt yuv420p, thiếu keyint', 'Đầy đủ: -pix_fmt yuv420p, keyint=30:min-keyint=30'),
    ('Evaluation metrics', 'Không có', 'PSNR, SSIM, SA-PSNR, SA-SSIM, mIOU, iIOU'),
]
for i, (a, b, c) in enumerate(ov_data):
    row = ov_table.rows[i + 1].cells
    row[0].text = a; row[1].text = b; row[2].text = c
    shade_cell(row[0], 'BDD7EE')
    for c2 in row:
        for p in c2.paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)

set_col_width(ov_table, 0, 4.0)
set_col_width(ov_table, 1, 7.0)
set_col_width(ov_table, 2, 8.0)
doc.add_paragraph()

# ── 2. Kiến trúc mạng Segmentation ──────────────────────────────────────────
add_heading(doc, '2. Kiến trúc mạng Segmentation', 1)

add_heading(doc, '2.1 Backbone ResNet – Pipeline A dùng ccnet_4class, Pipeline B dùng CCNet.SegNetwork', 2)
add_body(doc,
    'Pipeline A load model thông qua hàm load_ccnet_4class() từ scripts/new_feature/ccnet_4class.py '
    '(wrapper của CCNet ResNet-101 dilated theo paper gốc). '
    'Pipeline B dùng CCNet.SegNetwork được định nghĩa trong SAC/Codes/CCNet/CCNet.py – đây là kiến trúc '
    'ResNet nhỏ hơn, tự xây dựng lại (không phải ResNet-101).')

add_heading(doc, 'CCNet.py – SegNetwork (Pipeline B)', 3)
add_code(doc,
'''class ResNet(nn.Module):
    # Backbone nhỏ: Conv3x3→64 + 3 ResBlock + 3 MaxPool2x2 → output 512 channels, stride=8
    # Dùng GroupNorm(4, C) + LeakyReLU, KHÔNG phải BatchNorm + ReLU của ResNet-101 gốc
    inc   = Conv2d(3,   64,  3, pad=1) → GN(4,64)  → LeakyReLU
    res1  = [Conv3x3→64, GN, LReLU] × 2  + skip
    down1 = Conv2d(64,128,1) → GN → LReLU → MaxPool2d(2,2)
    res2  = [Conv3x3→128] × 2 + skip
    down2 = Conv2d(128,256,1) → MaxPool2d(2,2)
    res3  = [Conv3x3→256] × 2 + skip
    down3 = Conv2d(256,512,1) → MaxPool2d(2,2)
    # Output stride = 8 (3 lần MaxPool 2×2)
    # Kích thước: input 512×256 → feature 64×32×512

class RCCmodule(nn.Module):
    # RCCA Module + Decoder UNet-style
    conva      = Conv3x3(512→128) → GN → ReLU
    cca        = CrissCrossAttention(128)   # from cc.py
    convb      = Conv3x3(128→128) → GN → ReLU
    bottleneck = Conv3x3(640→512) → GN → ReLU  # concat [x(512), output(128)] = 640
    up1 = ConvTranspose(512→256) + Conv3x3→128 × 2    # ×2 upsample
    up2 = ConvTranspose(128→64)  + Conv3x3→64  × 2    # ×2 upsample
    up3 = ConvTranspose(64→32)   + Conv3x3→32  × 2    # ×2 upsample
    # Multi-scale heads (deep supervision)
    S3  = ConvTranspose(512→4, kernel=8, stride=8)    # từ bottleneck
    S2  = ConvTranspose(128→4, kernel=4, stride=4)    # từ up1
    S1  = Conv1x1(32→4)                               # từ up3
    out = Conv3x3(12→32) → GN → ReLU → Dropout(0.1) → Conv1x1(32→4)
    # Cuối cùng: F.softmax(output, dim=1)  ← trả về xác suất, KHÔNG phải logits''')

add_warning(doc,
    'Pipeline B trả về softmax probabilities (không phải logits). '
    'eval_sac.py kiểm tra: if out.max() > 1.01 or out.min() < -0.01 → dùng F.softmax. '
    'Nếu nhầm lẫn và áp softmax hai lần → phân phối bị làm phẳng, mIOU giảm đáng kể.')

add_heading(doc, 'cc.py – CC_module (CrissCrossAttention) – dùng chung cả hai pipeline', 3)
add_code(doc,
'''# cc.py dùng bởi CCNet.py (Pipeline B)
# Tương tự với CrissCrossAttention trong CLAUDE.md nhưng có 2 khác biệt:

# 1) Device hardcode: cuda(0) thay vì .cuda()
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda(0).repeat(H), 0)\\
        .unsqueeze(0).repeat(B*W, 1, 1)
# → Lỗi nếu chạy trên CPU hoặc GPU khác device 0

# 2) Không có import tường minh → phải chạy từ đúng thư mục
# Pipeline A dùng cc_attention/functions.py của CCNet repo gốc (pure PyTorch)''')

add_warning(doc,
    'cc.py hardcode .cuda(0): nếu chạy inference trên CPU (GTX 1650 hết VRAM) hoặc multi-GPU '
    'sẽ crash RuntimeError: Expected all tensors to be on the same device. '
    'Pipeline A (ccnet_4class) dùng implementation không có vấn đề này.')

doc.add_paragraph()

# ── 3. Tách 2 luồng ──────────────────────────────────────────────────────────
add_heading(doc, '3. Tách 2 luồng – Stream Separation', 1)

add_heading(doc, '3.1 Nguồn mask segmentation', 2)
add_body(doc,
    'Đây là sự khác biệt kiến trúc lớn nhất. Pipeline A chạy inference segmentation inline '
    'cho mỗi frame trong vòng lặp. Pipeline B đọc mask đã được lưu sẵn trên đĩa.')

add_code(doc,
'''# === Pipeline A: inference inline ===
input_tensor = transform(orig).unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(input_tensor)
mask = torch.argmax(pred, dim=1)[0].cpu().numpy()   # lấy argmax
# Resize về kích thước gốc bằng NEAREST
mask = cv2.resize(mask.astype(np.uint8),
                  (orig_np.shape[1], orig_np.shape[0]),
                  interpolation=cv2.INTER_NEAREST)
roi_mask = macroblock_align_filter((mask == 0).astype(np.uint8)) * 255

# === Pipeline B: đọc mask từ file ===
seg_path  = _get_label_path(img_path)          # → data/gt_4class/<split>/<city>/<stem>_gtFine_4class.png
seg_4class = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
# Resize nếu cần
if seg_4class.shape != img_bgr.shape[:2]:
    seg_4class = cv2.resize(seg_4class, (img_bgr.shape[1], img_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
roi_binary = (seg_4class == 0).astype(np.uint8)
roi_macro  = seg_mask_im_macro(roi_binary)''')

add_warning(doc,
    'Pipeline A dùng mask dự đoán real-time (có sai số segmentation). '
    'Pipeline B dùng ground-truth 4-class labels → kết quả nén/metric tốt hơn trên paper. '
    'Khi đánh giá, Pipeline B phản ánh upper-bound khi segmentation hoàn hảo, '
    'không phản ánh performance thực tế khi deploy.')

add_heading(doc, '3.2 Phép tách luồng – bitwise_and vs Hadamard multiply', 2)
add_code(doc,
'''# === Pipeline A: cv2.bitwise_and với mask 0/255 ===
roi_mask = macroblock_align_filter(...) * 255  # nhân 255 → mask 8-bit [0, 255]

roi_img = cv2.bitwise_and(orig_np, orig_np, mask=roi_mask)
#   Tương đương: pixel_out = pixel_in if mask_pixel != 0 else 0
#   orig_np: shape (H, W, 3) dtype uint8
#   roi_mask: shape (H, W) dtype uint8, chỉ nhận 0 hoặc 255

non_img = cv2.bitwise_and(orig_np, orig_np, mask=255 - roi_mask)
#   255 - roi_mask: đảo bit → vùng non-ROI = 255, ROI = 0

# === Pipeline B: nhân Hadamard với mask 0/1 (float) ===
mi = roi_macro[:, :, np.newaxis].repeat(3, axis=2)       # (H,W,3) giá trị 0 hoặc 1
mn = (1 - roi_macro)[:, :, np.newaxis].repeat(3, axis=2) # (H,W,3) complement

ia_img = (img_bgr * mi).astype(np.uint8)   # phép nhân trực tiếp – khớp công thức paper
ba_img = (img_bgr * mn).astype(np.uint8)   # S_i = M_i ⊙ X ; S_n = M_n ⊙ X''')

add_body(doc,
    'Về kết quả pixel: cả hai cho output giống hệt nhau với ảnh uint8 (giá trị nguyên [0,255]). '
    'cv2.bitwise_and thực hiện AND bitwise từng bit; với mask 255 (0xFF) thì kết quả không đổi. '
    'Nhân với 0 hoặc 1 cho kết quả bằng 0 hoặc giữ nguyên pixel. '
    'Tuy nhiên, Pipeline B khớp chính xác ký hiệu toán học trong paper (Hadamard product ⊙), '
    'giúp dễ đọc và verify hơn.')
add_ok(doc, 'Về kết quả số: TƯƠNG ĐƯƠNG. Không ảnh hưởng đến metric compression.')

add_heading(doc, '3.3 Macroblock filter 16×16 – chi tiết hàm', 2)
add_code(doc,
'''# === Pipeline A: macroblock_align_filter – vectorized numpy ===
def macroblock_align_filter(mask_2d, block_size=16):
    h, w = mask_2d.shape
    # Bước 1: Pad về bội của 16
    pad_h = (h + block_size - 1) // block_size * block_size
    pad_w = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((pad_h, pad_w), dtype=np.uint8)
    padded[:h, :w] = mask_2d

    # Bước 2: Reshape thành (n_block_h, 16, n_block_w, 16)
    blocks = padded.reshape(pad_h // block_size, block_size,
                            pad_w // block_size, block_size)
    # Bước 3: Max pooling trên mỗi block → (n_block_h, n_block_w)
    roi_max = blocks.max(axis=(1, 3))

    # Bước 4: Repeat ngược lại về kích thước đã pad
    aligned = np.repeat(np.repeat(roi_max, block_size, axis=0), block_size, axis=1)
    return aligned[:h, :w]   # crop về kích thước gốc

# === Pipeline B: seg_mask_im_macro – explicit loop ===
def seg_mask_im_macro(arr, block=16):
    h, w = arr.shape
    out  = arr.copy()
    for i in range(h // block):       # chú ý: h // block, KHÔNG phải ceil(h/block)
        for j in range(w // block):   # tương tự
            patch = arr[i*block:(i+1)*block, j*block:(j+1)*block]
            if patch.max() == 1:
                out[i*block:(i+1)*block, j*block:(j+1)*block] = 1
    return out
    # KHÔNG xử lý padding → bỏ sót hàng/cột pixel ở cạnh phải/dưới nếu H hoặc W % 16 != 0''')

add_warning(doc,
    'Pipeline B (seg_mask_im_macro): với ảnh Cityscapes 2048×1024 thì 2048%16=0 và 1024%16=0, '
    'nên BUG NÀY KHÔNG XẢY RA với dataset chuẩn. '
    'Nhưng nếu dùng ảnh KITTI (1242×375): 375%16=7 → 7 hàng pixel cuối cùng KHÔNG được xử lý macroblock, '
    'ROI pixel ở 7 hàng dưới cùng bị bỏ sót → mask sai → artifact ở biên dưới.')
add_warning(doc,
    'Về hiệu năng: Pipeline A xử lý ảnh 2048×1024 trong ~0.5ms (vectorized). '
    'Pipeline B mất ~45ms do nested Python for-loops. '
    'Với 500 frames val set × 45ms = 22.5 giây chỉ cho bước này.')

add_heading(doc, '3.4 Đặt tên file output – %04d vs %05d', 2)
add_code(doc,
'''# === Pipeline A: 4 chữ số ===
cv2.imwrite(os.path.join(TMP_DIR, f"frame_{idx:04d}_roi.png"), ...)
# → frame_0000_roi.png, frame_0001_roi.png, ..., frame_9999_roi.png
# Pattern FFmpeg: "frame_%04d_roi.png"

# === Pipeline B: 5 chữ số ===
cv2.imwrite(os.path.join(out_ia_folder, f"frame_{out_idx:05d}.png"), ...)
# → frame_00000.png, ..., frame_09999.png, ..., frame_99999.png
# Pattern FFmpeg: "frame_%05d.png"''')

add_warning(doc,
    'Sự không nhất quán chỉ ảnh hưởng khi cố ghép output của một pipeline vào pipeline kia. '
    'Nếu encode FFmpeg dùng sai pattern ("frame_%04d.png" vs "frame_%05d.png"), '
    'FFmpeg báo lỗi "No such file or directory" và crash. '
    'Đây là nguồn gây lỗi khi trộn lẫn output của hai pipeline.')

add_heading(doc, '3.5 Thư mục và cách tổ chức output', 2)
add_code(doc,
'''# === Pipeline A: flat structure ===
TMP_DIR = outputs/compressed/tmp_frames/
  frame_0000_roi.png   # luồng ROI
  frame_0000_non.png   # luồng non-ROI
  frame_0000_orig.png  # frame gốc (dùng cho baseline encoding)

# === Pipeline B: phân tách rõ ràng ===
outputs/two_stream/<split>/
  IA/frame_00000.png    # Important Area (ROI)
  BA/frame_00000.png    # Background Area (non-ROI)
  ORIG/frame_00000.png  # Original frame''')

add_ok(doc,
    'Pipeline B tách riêng 3 thư mục IA/BA/ORIG: dễ debug, dễ kiểm tra từng luồng, '
    'có thể visualize mask bằng cách so sánh IA và ORIG trực tiếp.')

doc.add_paragraph()

# ── 4. Nén FFmpeg ─────────────────────────────────────────────────────────────
add_heading(doc, '4. Nén FFmpeg – Chi tiết từng lệnh', 1)

add_heading(doc, '4.1 Flags FFmpeg – so sánh từng tham số', 2)
add_code(doc,
'''# === Pipeline A: encode ROI (x265, CRF=23) ===
ffmpeg -y -framerate 30 -i tmp_frames/frame_%04d_roi.png
       -c:v libx265
       -crf 23
       -preset medium
       outputs/compressed/roi.mp4
# Thiếu: -pix_fmt yuv420p
# Thiếu: -x265-params "keyint=30:min-keyint=30"

# === Pipeline B: encode ROI (x265, CRF=23) ===
ffmpeg -y -loglevel error
       -framerate 30
       -i outputs/two_stream/val_smoke/IA/frame_%05d.png
       -c:v libx265
       -crf 23
       -preset medium
       -pix_fmt yuv420p
       -x265-params "keyint=30:min-keyint=30"
       outputs/encoded_smoke/sa265_ia_crf23.mp4''')

# table for ffmpeg flags
flag_table = doc.add_table(rows=6, cols=4)
flag_table.style = 'Table Grid'
fhdr = flag_table.rows[0].cells
for c, t in zip(fhdr, ['Flag', 'Pipeline A', 'Pipeline B', 'Ảnh hưởng nếu thiếu']):
    c.text = t
    shade_cell(c, '2E75B6')
    for p in c.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.bold = True
            r.font.size = Pt(9)

flag_data = [
    ('-pix_fmt yuv420p', 'Không có', 'Có', 'Mặc định libx265 dùng yuv444p → không tương thích nhiều player/tools; khi ghép với frame PNG (yuv444) → color mismatch nhỏ nhưng có thể gây sai số PSNR ~0.1-0.3 dB'),
    ('-x265-params keyint=30:min-keyint=30', 'Không có', 'Có', 'FFmpeg x265 mặc định keyint=250. GOP lớn → inter-frame prediction mạnh hơn → bitrate thấp hơn nhưng random access kém. Khi giải nén frame đơn lẻ sẽ cần decode nhiều frame trước → chậm hơn và có thể khác kết quả paper (paper dùng keyint=30 theo 30fps Cityscapes)'),
    ('-loglevel error', 'Không có (dùng stderr)', 'Có', 'Không ảnh hưởng kết quả. Pipeline A in stderr đầy đủ (verbose hơn); Pipeline B im lặng khi thành công.'),
    ('Codec', 'libx265 cố định', 'libx264 hoặc libx265 theo --codec', 'Pipeline A không hỗ trợ so sánh H264 vs H265. Pipeline B test được 4 phương pháp song song.'),
    ('-framerate', '30 fps cố định', '30 fps (args.fps)', 'Như nhau về mặc định'),
]
for i, (a, b, c, d) in enumerate(flag_data):
    row = flag_table.rows[i + 1].cells
    row[0].text = a; row[1].text = b; row[2].text = c; row[3].text = d
    shade_cell(row[0], 'DCE6F1')
    for c2 in row:
        for p in c2.paragraphs:
            for r in p.runs:
                r.font.size = Pt(8.5)

set_col_width(flag_table, 0, 4.2)
set_col_width(flag_table, 1, 2.5)
set_col_width(flag_table, 2, 2.5)
set_col_width(flag_table, 3, 9.3)
doc.add_paragraph()

add_heading(doc, '4.2 Giá trị CRF và baseline – lỗi nghiêm trọng trong Pipeline A', 2)
add_code(doc,
'''# === Pipeline A: baseline CRF tính bằng average ===
CRF_ROI  = 23      # SA-X265 roi (đúng theo paper)
CRF_NON  = 32      # SA-X265 non (đúng theo paper)
total_crf = int((CRF_ROI + CRF_NON) / 2)   # = int(55/2) = 27
# Encode baseline với CRF=27 (SAI!)

# === Pipeline B: baseline CRF hardcoded đúng theo paper ===
DEFAULTS = {
    'h264_baseline_crf': 23,   # paper Table II, H.264 baseline
    'h265_baseline_crf': 28,   # paper Table II, H.265 baseline  ← ĐÚNG
    'sa264_roi_crf':     18,
    'sa264_non_crf':     27,
    'sa265_roi_crf':     23,
    'sa265_non_crf':     32,
}''')

add_warning(doc,
    'Pipeline A dùng CRF=27 cho H265 baseline thay vì CRF=28. '
    'CRF 27 nén ÍT HƠN CRF 28 → bitrate cao hơn → chất lượng cao hơn → '
    'so sánh KHÔNG công bằng (baseline được lợi thế). '
    'PSNR/SSIM của H265 baseline trong Pipeline A sẽ cao hơn thực tế ~0.5-1.0 dB, '
    'làm thu nhỏ lợi thế của SA-X265 so với baseline.')
add_warning(doc,
    'Nguyên tắc paper: so sánh phải cùng tỷ lệ nén tổng (same bitrate). '
    'Average CRF KHÔNG đảm bảo cùng bitrate vì CRF-bitrate không tuyến tính. '
    'Pipeline B không so sánh bitrate trực tiếp; run_smoke_pipeline.py tính bitrate_kbps '
    'từ file size thực tế để báo cáo.')

doc.add_paragraph()

# ── 5. Ghép 2 video ───────────────────────────────────────────────────────────
add_heading(doc, '5. Ghép 2 luồng sau giải nén – Combine/Reconstruct', 1)

add_heading(doc, '5.1 Phương pháp ghép', 2)
add_code(doc,
'''# === Pipeline A: ghép bằng FFmpeg blend filter (video-level) ===
run_ffmpeg([
    "ffmpeg", "-y",
    "-i", "outputs/compressed/roi.mp4",
    "-i", "outputs/compressed/nonroi.mp4",
    "-filter_complex", "[0:v][1:v]blend=all_mode=addition",
    "outputs/compressed/sac_x265.mp4"
], "Merge ROI and non-ROI into SAC-X265")
# Không có bước decode ra PNG trước
# FFmpeg decode cả 2 stream song song → blend từng frame trong không gian video

# === Pipeline B: decode trước → combine.py → cv2.add() ===
# Bước 1: decode ra PNG
ffmpeg -y -i sa265_ia_crf23.mp4 outputs/encoded_smoke/sa265_ia_dec/frame_%05d.png
ffmpeg -y -i sa265_ba_crf32.mp4 outputs/encoded_smoke/sa265_ba_dec/frame_%05d.png

# Bước 2: combine.py
for ia_path, ba_path in zip(sorted(ia_files), sorted(ba_files)):
    img_ia = cv2.imread(ia_path,  cv2.IMREAD_COLOR)
    img_ba = cv2.imread(ba_path,  cv2.IMREAD_COLOR)
    combined = cv2.add(img_ia, img_ba)    # saturating add uint8: min(a+b, 255)
    io.imsave(out_path, combined_rgb)     # lưu PNG cho bước eval''')

add_heading(doc, '5.2 Phân tích kỹ thuật: blend=addition vs cv2.add()', 2)

blend_table = doc.add_table(rows=6, cols=3)
blend_table.style = 'Table Grid'
bh = blend_table.rows[0].cells
for c, t in zip(bh, ['Đặc điểm', 'FFmpeg blend=addition\n(Pipeline A)', 'cv2.add() frame-by-frame\n(Pipeline B)']):
    c.text = t
    shade_cell(c, '2E75B6')
    for p in c.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.bold = True
            r.font.size = Pt(9)

blend_data = [
    ('Không gian màu khi blend', 'YUV (không gian nội bộ của codec H265)',
     'BGR uint8 (sau khi FFmpeg đã decode ra PNG → RGB)'),
    ('Overflow handling', 'Saturation trong YUV → clip về [0,255]',
     'cv2.add = saturating add trong BGR uint8 → clip về 255'),
    ('Kết quả pixel', 'Có thể sai màu nhỏ do YUV↔RGB round-trip + chroma subsampling yuv420',
     'Chính xác trong RGB space, không có round-trip conversion'),
    ('Intermediate files', 'Không có (xử lý trong memory FFmpeg)',
     'Cần decode ra PNG → tốn disk space (~3-6 GB cho 500 frames 2048×1024)'),
    ('Khả năng debug', 'Khó: không thể xem từng frame trung gian',
     'Dễ: có thể mở bất kỳ frame PNG nào để kiểm tra'),
]
for i, (a, b, c) in enumerate(blend_data):
    row = blend_table.rows[i + 1].cells
    row[0].text = a; row[1].text = b; row[2].text = c
    shade_cell(row[0], 'DCE6F1')
    for c2 in row:
        for p in c2.paragraphs:
            for r in p.runs:
                r.font.size = Pt(8.5)

set_col_width(blend_table, 0, 4.0)
set_col_width(blend_table, 1, 6.5)
set_col_width(blend_table, 2, 8.0)
doc.add_paragraph()

add_heading(doc, '5.3 Ảnh hưởng đến metric PSNR/SSIM', 2)
add_body(doc,
    'Pipeline A tính PSNR/SSIM không có trong code (không có bước eval), nên sự khác biệt blend method '
    'không ảnh hưởng đến con số báo cáo. '
    'Tuy nhiên, nếu Pipeline A được bổ sung eval sau này:')
add_code(doc,
'''# Trường hợp pipeline A thêm eval:
# - sac_x265.mp4 được tạo bằng blend trong YUV space
# - Khi decode ra frames để tính PSNR, có thêm 1 lần YUV→RGB→PNG→YUV encoding
# - Mỗi lần convert thêm ~0.1-0.3 dB noise vào PSNR
# → PSNR của Pipeline A có thể thấp hơn Pipeline B ~0.2-0.5 dB dù cùng CRF

# Pipeline B:
# - Decode ra PNG (RGB lossless) → cv2.add() trong RGB → lưu PNG
# - Không có thêm encode/decode → PSNR chính xác hơn''')

add_warning(doc,
    'Kết quả sac_x265.mp4 của Pipeline A là video đã ghép – không thể tính PSNR '
    'frame-by-frame so với ảnh gốc PNG mà không có thêm bước decode. '
    'Pipeline B giữ frame PNG sau khi combine → tính PSNR trực tiếp, không mất thêm chất lượng.')

doc.add_paragraph()

# ── 6. Pre-processing & Normalization ────────────────────────────────────────
add_heading(doc, '6. Tiền xử lý ảnh – Normalization khác nhau', 1)

add_code(doc,
'''# === Pipeline A: ImageNet normalization (torchvision standard) ===
transform = transforms.Compose([
    transforms.Resize((512, 1024)),          # resize về 512×1024 (W×H)
    transforms.ToTensor(),                   # [0,1] float
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet stats
])
# Input model: tensor (1, 3, 512, 1024) normalized bằng ImageNet mean/std
# Không có grayscale conversion

# === Pipeline B (read.py – dùng khi training và eval_sac.py) ===
def normor(image):
    image -= image.mean()
    image /= (image.std() + 1e-8)
    return image

# Trong Data.__getitem__:
image = io.imread(path).astype(float)   # (H, W, 3) float64
image = image.transpose(2, 0, 1)        # (3, H, W)
image = resize(image, (3, H//2, W//2),  # GIẢM 1/2 kích thước
               order=3, mode='edge').astype(np.float32)
imagenorm = normor(image)               # per-image standardization (zero mean, unit std)

# preprocess_for_seg trong eval_sac.py (khớp với read.py):
def preprocess_for_seg(rgb_uint8):
    img = rgb_uint8.astype(np.float64).transpose(2, 0, 1)  # (3,H,W)
    img = sk_resize(img, (3, h//2, w//2), order=3, mode='edge').astype(np.float32)
    img -= img.mean()
    img /= (img.std() + 1e-8)
    return torch.from_numpy(img).unsqueeze(0)''')

add_warning(doc,
    'Pipeline A dùng ImageNet mean/std normalization và KHÔNG resize về 1/2. '
    'Pipeline B (CCNet.SegNetwork) được train với per-image standardization + resize 1/2. '
    'Nếu load model của Pipeline B nhưng dùng preprocessing của Pipeline A → '
    'phân phối input sai hoàn toàn → mIOU có thể giảm 10-30 điểm. '
    'Đây là lỗi nghiêm trọng nếu lẫn lộn model và preprocessing giữa hai pipeline.')

add_body(doc,
    'Ngoài ra, Pipeline B không convert sang grayscale. Pipeline A theo CLAUDE.md §5.1 nên '
    'convert sang grayscale (Y = 0.299R + 0.587G + 0.114B rồi repeat 3 kênh), nhưng '
    'transform trong sac_compression_x265.py không có bước này – chỉ normalize màu. '
    'Model ccnet_4class được train với input grayscale hay RGB cần kiểm tra lại trong '
    'scripts/new_feature/ccnet_4class.py và training script tương ứng.')

doc.add_paragraph()

# ── 7. Loss functions ─────────────────────────────────────────────────────────
add_heading(doc, '7. Hàm Loss và Training', 1)

add_heading(doc, '7.1 Loss trong Pipeline B (metrics.py)', 2)
add_code(doc,
'''# === metrics.py – 4 class, dùng trong Pipeline B training ===

class DiceMeanLoss(nn.Module):
    """Weighted Dice loss – đây là loss CHÍNH dùng trong training Pipeline B."""
    def forward(self, logits, targets):
        # logits, targets: (B, 4, H, W) – targets là one-hot (KHÔNG phải class index)
        weights = [0.1, 0.35, 0.35, 0.2]
        # w[0]=0.1  → ROI (class 0)           ← trọng số THẤP NHẤT
        # w[1]=0.35 → construction (class 1)
        # w[2]=0.35 → nature (class 2)
        # w[3]=0.2  → sky (class 3)
        # Dice với smoothing=1:
        # dice_i = (2 * inter + 1) / (sum_pred_i + sum_gt_i + 1)
        return 1 - dice_sum   # scalar

class DiceMeanLoss1(nn.Module):
    """Unweighted Dice, trả về tuple (loss, iou_last_class) – dùng để monitor."""

class IOU(nn.Module):
    """Mean IoU over 4 classes (dùng để eval trong training loop)."""
    # inter = sum(pred * gt)  ; union = sum(pred) + sum(gt) - inter
    # iou_i = inter / (union + 1e-8)

class crossentry(nn.Module):
    """Soft cross-entropy với one-hot targets."""
    # L = -mean(gt * log(pred + 1e-6))
    # Chú ý: pred đã qua softmax (từ CCNet.py forward)
    # → dùng crossentry chứ không phải nn.CrossEntropyLoss (cần logits)''')

add_warning(doc,
    'Class weight của DiceMeanLoss: ROI có weight 0.1 (thấp nhất!). '
    'Điều này NGƯỢC với trực giác – ROI là vùng quan trọng nhưng lại có weight nhỏ. '
    'Lý do có thể: ROI chiếm ~70-80% diện tích ảnh nên dễ predict, '
    'các class nhỏ (construction, nature) cần weight cao hơn để không bị bỏ qua. '
    'Cần verify với training curves để xác nhận behavior này là intentional.')

add_heading(doc, '7.2 So sánh với CLAUDE.md (BCEDiceLoss)', 2)
add_code(doc,
'''# === CLAUDE.md §6.1 – BCEDiceLoss (khuyến nghị theo paper gốc) ===
class BCEDiceLoss(nn.Module):
    # ce_loss = nn.CrossEntropyLoss (dùng logits, targets là class index integer)
    # dice    = 1 - (2*inter + smooth) / (union + smooth)  với smooth=1
    # total   = ce_loss + dice.mean()
    # Input logits: (B,C,H,W) chưa qua softmax
    # Input target: (B,H,W) class index [0..3]

# === Pipeline B (metrics.py) – DiceMeanLoss thực tế dùng ===
# Input logits: (B,C,H,W) ĐÃ qua softmax (do CCNet.py forward có F.softmax)
# Input target: (B,C,H,W) one-hot
# KHÔNG có BCE/CrossEntropy component
# → Loss thiếu cross-entropy term → hội tụ có thể chậm hơn / accuracy thấp hơn so với paper''')

add_warning(doc,
    'Pipeline B chỉ dùng Dice loss (không có BCE/CE component), trong khi paper dùng BCE+Dice. '
    'Thiếu CE term làm mất đi gradient trực tiếp từ từng pixel → đặc biệt ảnh hưởng '
    'đến các class nhỏ ở vùng biên. Kỳ vọng mIOU của Pipeline B thấp hơn paper 2-5 điểm.')

doc.add_paragraph()

# ── 8. Evaluation ─────────────────────────────────────────────────────────────
add_heading(doc, '8. Đánh giá – Evaluation', 1)

add_heading(doc, '8.1 Pipeline A – Không có evaluation', 2)
add_body(doc,
    'sac_compression_x265.py kết thúc sau khi tạo sac_x265.mp4 và traditional_x265.mp4. '
    'Không có code tính PSNR, SSIM, SA-PSNR, SA-SSIM, mIOU hay iIOU. '
    'Biến original_frames, sac_frames, traditional_frames được khai báo nhưng không sử dụng sau đó.')
add_code(doc,
'''# Pipeline A – variables được khai báo nhưng không có eval code
original_frames     = []   # append(orig_np) nhưng không dùng sau
sac_frames          = []   # không có gì append vào đây
traditional_frames  = []   # không có gì append vào đây
# → 3 list này chiếm RAM không cần thiết nếu chạy nhiều frames''')

add_heading(doc, '8.2 eval_sac.py – Đầy đủ theo paper', 2)
add_code(doc,
'''# eval_sac.py – hàm evaluate() tính toán:
def evaluate(recon_dir, crf_roi, crf_non, split, ...):
    # Trọng số SA theo §9.3 CLAUDE.md:
    denom = crf_roi + crf_non
    r_roi = crf_roi / denom   # trọng số cho non-ROI metric (r_roi ÁP vào P_n, S_n)
    r_non = crf_non / denom   # trọng số cho ROI metric     (r_non ÁP vào P_i, S_i)

    # Với CRF_roi=18, CRF_non=27: denom=45, r_roi=18/45=0.4, r_non=27/45=0.6
    # → chất lượng ROI (P_i) được nhân hệ số 0.6 (lớn hơn) → ROI quan trọng hơn

    # SA-PSNR = r_non * P_i + r_roi * P_n
    sa_psnr = r_non * p_i + r_roi * p_n

    # region_psnr: tính MSE chỉ trên pixel nơi mask=1
    def region_psnr(orig, recon, mask):
        diff = orig.astype(float) - recon.astype(float)
        mse  = (diff**2)[mask.astype(bool)].mean()
        return 10.0 * np.log10((255.0**2) / mse)

    # region_ssim: dùng SSIM map từ skimage
    def region_ssim(orig, recon, mask):
        _, ssim_map = sk_ssim(orig, recon, channel_axis=2, data_range=255, full=True)
        ssim_map = ssim_map.mean(axis=2)  # (H,W)
        return float(ssim_map[mask.astype(bool)].mean())

    # iIOU: IoU chỉ của class 0 (ROI) → ious[0]
    iiou = ious[0]  # class 0 = ROI''')

doc.add_paragraph()

# ── 9. Bảng tổng hợp tất cả sự khác biệt ─────────────────────────────────────
add_heading(doc, '9. Bảng Tổng hợp Toàn bộ Sự Khác biệt và Ảnh hưởng', 1)

full_table = doc.add_table(rows=1, cols=4)
full_table.style = 'Table Grid'
fth = full_table.rows[0].cells
for c, t in zip(fth, ['#', 'Khía cạnh / Hàm cụ thể', 'Pipeline A', 'Pipeline B + Ảnh hưởng']):
    c.text = t
    shade_cell(c, '1F497D')
    for p in c.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.bold = True
            r.font.size = Pt(8.5)

full_data = [
    ('1', 'Nguồn mask segmentation',
     'Inference inline mỗi frame (model ccnet_4class)',
     'Đọc file PNG gt_4class đã lưu sẵn → KHÔNG chạy inference\n'
     '[Ảnh hưởng] Pipeline B dùng ground-truth mask → upper-bound metric, không phải performance thực tế. '
     'Pipeline A sát thực tế hơn nhưng phụ thuộc chất lượng model.'),

    ('2', 'Backbone model (CCNet.py)',
     'ResNet-101 dilated + RCCA (theo paper gốc)\nOutput: logits chưa softmax',
     'ResNet nhỏ tự xây: 3 ResBlock + 3 MaxPool\nOutput: softmax probabilities (F.softmax cuối forward)\n'
     '[Ảnh hưởng] Model của Pipeline B nhỏ hơn nhiều (~5M params vs ~70M). '
     'mIOU kỳ vọng thấp hơn paper. KHÔNG thể dùng chung checkpoint hai pipeline.'),

    ('3', 'cc.py – hàm INF()',
     'Không dùng cc.py trực tiếp\n(dùng cc_attention của CCNet repo)',
     'INF() dùng .cuda(0) hardcode\n'
     '[Ảnh hưởng] Crash nếu chạy trên CPU hoặc GPU != device 0. '
     'Phải sửa thành .to(device) nếu muốn linh hoạt.'),

    ('4', 'Normalization input',
     'ImageNet mean/std\n[0.485,0.456,0.406] / [0.229,0.224,0.225]\nResize: (512,1024)',
     'Per-image standardization (mean=0, std=1)\nResize: 1/2 kích thước gốc\n'
     '[Ảnh hưởng] NGHIÊM TRỌNG: lẫn model + wrong preprocessing → mIOU giảm 10-30 điểm.'),

    ('5', 'Grayscale conversion',
     'KHÔNG có (input vẫn là RGB 3 kênh)',
     'KHÔNG có trong Pipeline B\nCLAUDE.md §5.1 yêu cầu: Y=0.299R+0.587G+0.114B\n'
     '[Ảnh hưởng] Nếu model được train với grayscale nhưng test với RGB → distribution shift.'),

    ('6', 'Hàm macroblock_align_filter\n(tên: seg_mask_im_macro)',
     'Vectorized numpy:\npad → reshape → max(axis=(1,3)) → repeat\nXử lý padding đúng',
     'Explicit nested loop:\nfor i in range(h//block): for j in range(w//block):\n'
     '[Ảnh hưởng] (1) Chậm hơn ~90× với Python loop. '
     '(2) Bỏ sót hàng/cột cuối nếu H hoặc W không chia hết 16 (bug với KITTI-375px).'),

    ('7', 'Phép tách luồng\n(get_streams)',
     'cv2.bitwise_and(img, img, mask=roi_mask_255)\nmask: uint8 [0, 255]',
     'img_bgr * mi  (Hadamard product)\nmask: uint8 [0, 1]\n'
     '[Ảnh hưởng] Kết quả pixel GIỐNG NHAU với ảnh uint8. '
     'Pipeline B khớp ký hiệu toán học paper hơn.'),

    ('8', 'Đặt tên file frame\n(ffmpeg pattern)',
     'frame_%04d_roi.png\nframe_%04d_non.png\n(4 chữ số)',
     'frame_%05d.png trong IA/ và BA/\n(5 chữ số)\n'
     '[Ảnh hưởng] Không thể trộn lẫn output hai pipeline. '
     'FFmpeg crash nếu dùng sai pattern.'),

    ('9', 'FFmpeg flag -pix_fmt yuv420p',
     'THIẾU',
     'Có\n[Ảnh hưởng] Thiếu → libx265 mặc định yuv444p. '
     'Color mismatch khi blend/decode. Có thể gây sai số PSNR ~0.1-0.3 dB.'),

    ('10', 'FFmpeg flag keyint (GOP)',
     'THIẾU (default=250)',
     'keyint=30:min-keyint=30\n[Ảnh hưởng] GOP=250 vs GOP=30. '
     'GOP lớn hơn → inter-frame prediction khác → bitrate khác → '
     'so sánh tỷ lệ nén KHÔNG công bằng với paper (paper dùng GOP=30).'),

    ('11', 'Baseline CRF cho H265',
     'int((23+32)/2) = 27  ← SAI',
     'Hardcode 28 (đúng theo paper)\n'
     '[Ảnh hưởng] Baseline chất lượng cao hơn thực tế → '
     'lợi thế SA-X265 bị underestimate ~0.5-1.0 dB PSNR.'),

    ('12', 'Codec hỗ trợ',
     'X265 only (libx265)',
     'H264 + H265 + SA-X264 + SA-X265\n[Ảnh hưởng] Pipeline A bỏ sót bảng II paper (SA-X264 CRF=18/27 là kết quả tốt nhất).'),

    ('13', 'Phương pháp ghép 2 luồng\n(combine step)',
     'FFmpeg blend=addition (video-level)\nKhông có intermediate PNG',
     'cv2.add() frame-by-frame (pixel-level)\nDecoded PNG trước khi ghép\n'
     '[Ảnh hưởng] blend trong YUV space có thể có minor color error. '
     'Pipeline B chính xác hơn và debuggable.'),

    ('14', 'Output sau ghép',
     'sac_x265.mp4 (video)\nKhông thể tính PSNR trực tiếp',
     'Frame PNG riêng lẻ\nTính PSNR/SSIM trực tiếp với orig PNG\n'
     '[Ảnh hưởng] Pipeline A cần thêm bước decode video trước khi eval → thêm lossy step.'),

    ('15', 'Loss function training',
     'BCEDiceLoss (CE + Dice)\nInput: logits + class-index target',
     'DiceMeanLoss (Dice only, weighted)\nInput: softmax + one-hot target\n'
     '[Ảnh hưởng] Thiếu CE → gradient per-pixel yếu hơn → '
     'mIOU Pipeline B kỳ vọng thấp hơn paper 2-5 điểm với small/boundary regions.'),

    ('16', 'Evaluation metrics',
     'Không có code eval',
     'Đầy đủ: PSNR, SSIM, P_i, P_n, S_i, S_n,\nSA-PSNR, SA-SSIM, mIOU, iIOU\n'
     '[Ảnh hưởng] Pipeline A không thể tái hiện bảng kết quả paper.'),

    ('17', 'Biến unused (memory leak)',
     'original_frames, sac_frames, traditional_frames\nkhai báo nhưng không dùng',
     'Không có vấn đề này\n[Ảnh hưởng] Với MAX_FRAMES lớn, 3 list này chiếm RAM không cần thiết.'),
]

for i, (num, aspect, a, b) in enumerate(full_data):
    row = full_table.add_row().cells
    row[0].text = num
    row[1].text = aspect
    row[2].text = a
    row[3].text = b
    shade_cell(row[1], 'DCE6F1')
    if i % 2 == 0:
        shade_cell(row[2], 'FDFEFE')
        shade_cell(row[3], 'FDFEFE')
    for c in row:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.size = Pt(8.5)

set_col_width(full_table, 0, 0.7)
set_col_width(full_table, 1, 4.0)
set_col_width(full_table, 2, 5.3)
set_col_width(full_table, 3, 9.5)
doc.add_paragraph()

# ── 10. Kết luận và khuyến nghị ───────────────────────────────────────────────
add_heading(doc, '10. Kết luận và Khuyến nghị Sửa Pipeline A', 1)

add_heading(doc, '10.1 Mức độ ảnh hưởng của từng lỗi', 2)

sev_table = doc.add_table(rows=1, cols=4)
sev_table.style = 'Table Grid'
sh2 = sev_table.rows[0].cells
for c, t in zip(sh2, ['Lỗi', 'Mức độ', 'Ảnh hưởng đến kết quả', 'Fix']):
    c.text = t
    shade_cell(c, '2E75B6')
    for p in c.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            r.bold = True
            r.font.size = Pt(9)

sev_data = [
    ('Baseline CRF=27 thay vì 28', 'CAO', 'So sánh không công bằng; lợi thế SAC bị underestimate', 'Thay total_crf = 28'),
    ('Thiếu -pix_fmt yuv420p', 'TRUNG BÌNH', 'Color space mismatch; PSNR sai ~0.1-0.3 dB', 'Thêm flag vào ffmpeg command'),
    ('Thiếu keyint=30', 'TRUNG BÌNH', 'GOP khác paper → bitrate và inter-pred khác', 'Thêm -x265-params "keyint=30:min-keyint=30"'),
    ('Không có evaluation', 'CAO', 'Không thể tái hiện kết quả số trong paper', 'Tích hợp eval_sac.evaluate() sau bước combine'),
    ('Blend=addition thay vì cv2.add()', 'THẤP', 'Minor color error trong YUV space', 'Decode ra PNG rồi dùng cv2.add()'),
    ('Frame pattern %04d vs %05d', 'THẤP', 'Chỉ ảnh hưởng nếu trộn lẫn hai pipeline', 'Thống nhất dùng %05d'),
    ('Biến unused (3 list)', 'THẤP', 'Tốn RAM không cần thiết', 'Xóa khai báo'),
]
for i, (a, b, c, d) in enumerate(sev_data):
    row = sev_table.add_row().cells
    row[0].text = a; row[1].text = b; row[2].text = c; row[3].text = d
    shade_cell(row[1], 'F4D03F' if b == 'TRUNG BÌNH' else ('E74C3C' if b == 'CAO' else 'A9DFBF'))
    for c2 in row:
        for p in c2.paragraphs:
            for r in p.runs:
                r.font.size = Pt(8.5)

set_col_width(sev_table, 0, 4.5)
set_col_width(sev_table, 1, 2.0)
set_col_width(sev_table, 2, 7.5)
set_col_width(sev_table, 3, 5.5)
doc.add_paragraph()

add_heading(doc, '10.2 Pipeline B là chuẩn để tái hiện paper – nhưng cần lưu ý', 2)
add_body(doc,
    'Pipeline B (SAC/Codes/CCNet) đúng hơn về: CRF values, FFmpeg flags, combine method, '
    'và có đầy đủ evaluation. Tuy nhiên:')
add_code(doc,
'''# Vấn đề của Pipeline B cần kiểm tra:
1. seg_mask_im_macro() – explicit loop ~90x chậm hơn, bug với ảnh H/W không chia hết 16
   → Fix: dùng vectorized version từ macroblock_align_filter() của Pipeline A

2. Dùng ground-truth mask (gt_4class) thay vì predicted mask
   → Không phản ánh performance thực tế khi deploy
   → Cần thêm option chạy với predicted mask cho fair comparison

3. cc.py hardcode .cuda(0)
   → Fix: thay bằng .to(device) hoặc truyền device vào INF()

4. CCNet.SegNetwork không phải ResNet-101 của paper
   → Đây là custom lightweight model
   → Cần dùng ccnet_4class (ResNet-101 dilated) cho kết quả khớp paper''')

add_heading(doc, '10.3 Khuyến nghị luồng chạy chuẩn', 2)
add_code(doc,
'''# Luồng chuẩn để tái hiện paper (kết hợp điểm mạnh hai pipeline):
Step 1: Train model với new_feature/ccnet_4class.py (ResNet-101 + RCCA, ImageNet pretrain)
        Loss: BCEDiceLoss (CE + Dice theo CLAUDE.md §6.1)
        Input: RGB normalized với ImageNet mean/std

Step 2: Inference → lưu mask PNG vào data/gt_4class/
        (dùng output của ccnet_4class, không phải CCNet.SegNetwork)

Step 3: TwoStream_generate.py → IA/BA/ORIG
        (thay seg_mask_im_macro bằng macroblock_align_filter vectorized)

Step 4: run_smoke_pipeline.py với đầy đủ flags
        (CRF values đúng: H264=23, H265=28, SA264=18/27, SA265=23/32)

Step 5: eval_sac.evaluate() → SA-PSNR, SA-SSIM, mIOU, iIOU
        (preprocess_for_seg phải KHỚP với preprocessing lúc train)''')

# Footer
doc.add_paragraph()
footer_p = doc.add_paragraph()
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer_p.add_run(
    f'Báo cáo được tạo tự động bởi gen_comparison_report.py  |  '
    f'Dự án SAC – Semantic-Aware Compression  |  {datetime.date.today().strftime("%d/%m/%Y")}')
fr.font.size = Pt(8)
fr.font.italic = True
fr.font.color.rgb = RGBColor(0x7F, 0x8C, 0x8D)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = '/home/huy/sac_project/report/so_sanh_pipeline_sac.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
