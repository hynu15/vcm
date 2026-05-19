"""Sweep nhiều cặp CRF/QP để vẽ RD curve bitrate vs mIoU.

Tái dùng các bước của run_smoke_pipeline.py + eval_sac.evaluate(...) để chạy
4 phương pháp (H.264, H.265, SA-X264, SA-X265) ở 4 điểm CRF khác nhau,
tổng 16 điểm RD. Sau đó vẽ matplotlib bitrate (kbps) vs mIoU.

Two-stream (IA / BA / ORIG) chỉ được sinh MỘT lần cho toàn bộ sweep.

Ví dụ:
  python scripts/run_rd_curve_miou.py --num-frames 10 --suffix _rd10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE_SCRIPT = Path(__file__).resolve()
PROJECT     = HERE_SCRIPT.parents[1]
CCNET_DIR   = PROJECT / 'scripts/Semantic-Aware-Video-Compression-for-Automotive-Cameras/Codes/CCNet'
OUTPUTS     = PROJECT / 'outputs'

DEFAULT_SWEEP = {
    'h264':  [(15, 15), (22, 22), (30, 30), (38, 38)],
    'h265':  [(20, 20), (27, 27), (34, 34), (40, 40)],
    'sa264': [(10, 19), (17, 26), (25, 34), (33, 42)],
    'sa265': [(15, 24), (22, 31), (29, 38), (35, 44)],
}
CODEC_OF = {'h264': '264', 'h265': '265', 'sa264': '264', 'sa265': '265'}


def sh(cmd, step):
    print(f"\n→ [{step}]  $ {' '.join(map(str, cmd))}")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError(f"step '{step}' failed (exit {r.returncode})")
    return r.stdout


def ffmpeg_encode(codec, framerate, in_pattern, crf, out_mp4, preset='medium', gop=30):
    enc = 'libx264' if codec == '264' else 'libx265'
    params = f'keyint={gop}:min-keyint={gop}'
    pkey   = '-x264-params' if codec == '264' else '-x265-params'
    sh(['ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(framerate), '-i', str(in_pattern),
        '-c:v', enc, '-crf', str(crf), '-preset', preset,
        '-pix_fmt', 'yuv420p', pkey, params, str(out_mp4)],
       f'encode {codec} crf={crf}')


def ffmpeg_decode(mp4, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sh(['ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(mp4), str(out_dir / 'frame_%05d.png')],
       f'decode {mp4.name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--num-frames',  type=int, default=10)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--split',       type=str, default='val',
                    choices=['train', 'val', 'test'])
    ap.add_argument('--suffix',      type=str, default='_rd')
    ap.add_argument('--model',       type=str, default='net_epoch_17-fuseNetwork.pkl')
    ap.add_argument('--arch',        type=str, default='ccnet',
                    choices=['ccnet', 'pidnet_l'])
    ap.add_argument('--fps',         type=int, default=30)
    ap.add_argument('--preset',      type=str, default='medium')
    ap.add_argument('--methods',     type=str, default='h264,h265,sa264,sa265')
    ap.add_argument('--out-dir',     type=str, default='',
                    help='thư mục output (mặc định outputs/rd_curve_miou{suffix})')
    ap.add_argument('--reuse-streams', action='store_true',
                    help='nếu IA/BA/ORIG đã tồn tại thì bỏ qua bước TwoStream_generate')
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for m in methods:
        if m not in DEFAULT_SWEEP:
            raise ValueError(f'method không hỗ trợ: {m}')

    out_dir = Path(args.out_dir) if args.out_dir else OUTPUTS / f'rd_curve_miou{args.suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) TwoStream (1 lần) ──────────────────────────────────────────────────
    t0 = time.time()
    split_dir = OUTPUTS / 'two_stream' / f'{args.split}{args.suffix}'
    ia_dir, ba_dir, orig_dir = split_dir / 'IA', split_dir / 'BA', split_dir / 'ORIG'

    have_streams = ia_dir.exists() and ba_dir.exists() and orig_dir.exists()
    if not (args.reuse_streams and have_streams):
        print(f"\n=== Step 1: tách 2 luồng + lưu ORIG cho {args.num_frames} frame ===")
        sh([sys.executable, str(CCNET_DIR / 'TwoStream_generate.py'),
            '--split', args.split,
            '--max-steps', str(args.num_frames),
            '--start', str(args.start_frame),
            '--out-suffix', args.suffix],
           'two-stream')
    else:
        print(f"\n[reuse] {split_dir} đã tồn tại, bỏ qua TwoStream_generate")

    # ── 2) Encode/Decode/Combine cho mọi (method, crf_roi, crf_non) ──────────
    enc_root = out_dir / 'encoded'
    enc_root.mkdir(parents=True, exist_ok=True)
    combined_root = out_dir / 'combined'
    combined_root.mkdir(parents=True, exist_ok=True)

    rd_points = []  # list of dict mô tả mỗi điểm RD
    for method in methods:
        codec = CODEC_OF[method]
        for crf_roi, crf_non in DEFAULT_SWEEP[method]:
            tag = f'{method}_r{crf_roi}_n{crf_non}'
            mp4s = []
            if method in ('h264', 'h265'):
                mp4 = enc_root / f'{tag}_full.mp4'
                ffmpeg_encode(codec, args.fps, orig_dir / 'frame_%05d.png', crf_roi, mp4,
                              preset=args.preset)
                dec_dir = enc_root / f'{tag}_dec'
                ffmpeg_decode(mp4, dec_dir)
                recon_dir = dec_dir
                mp4s.append(mp4)
            else:
                mp4_ia = enc_root / f'{tag}_ia.mp4'
                mp4_ba = enc_root / f'{tag}_ba.mp4'
                ffmpeg_encode(codec, args.fps, ia_dir / 'frame_%05d.png', crf_roi, mp4_ia,
                              preset=args.preset)
                ffmpeg_encode(codec, args.fps, ba_dir / 'frame_%05d.png', crf_non, mp4_ba,
                              preset=args.preset)
                ia_dec = enc_root / f'{tag}_ia_dec'
                ba_dec = enc_root / f'{tag}_ba_dec'
                ffmpeg_decode(mp4_ia, ia_dec)
                ffmpeg_decode(mp4_ba, ba_dec)

                recon_dir = combined_root / tag
                sh([sys.executable, str(CCNET_DIR / 'combine.py'),
                    '--split', f'{args.split}{args.suffix}',
                    '--codec', codec,
                    '--crf-roi', str(crf_roi),
                    '--crf-non', str(crf_non),
                    '--ia-dir', str(ia_dec),
                    '--ba-dir', str(ba_dec),
                    '--out-dir', str(recon_dir)],
                   f'combine {tag}')
                mp4s.extend([mp4_ia, mp4_ba])

            rd_points.append({
                'method': method, 'codec': codec,
                'crf_roi': crf_roi, 'crf_non': crf_non,
                'tag': tag, 'recon_dir': str(recon_dir),
                'mp4s': [str(p) for p in mp4s],
            })

    # ── 3) Evaluate (load seg model 1 lần) ───────────────────────────────────
    print("\n=== Step 3: evaluate metrics ===")
    sys.path.insert(0, str(CCNET_DIR))
    import eval_sac

    seg_model = None
    if args.model:
        seg_model = eval_sac.load_seg_model(args.model, args.arch, 4)
        print(f"[loaded] {args.model}  arch={args.arch}")

    rows = []
    for pt in rd_points:
        print(f"\n--- {pt['tag']} (CRF roi={pt['crf_roi']}, non={pt['crf_non']}) ---")
        r = eval_sac.evaluate(
            recon_dir=pt['recon_dir'],
            crf_roi=pt['crf_roi'], crf_non=pt['crf_non'],
            split=args.split, max_steps=args.num_frames,
            num_classes=4, verbose=False, seg_model=seg_model,
            start=args.start_frame,
        )
        total_bytes = sum(Path(p).stat().st_size for p in pt['mp4s'])
        duration    = args.num_frames / args.fps
        r['method']       = pt['method']
        r['tag']          = pt['tag']
        r['bitrate_kbps'] = (total_bytes * 8 / duration) / 1000.0
        r['file_size_kb'] = total_bytes / 1024.0
        rows.append(r)

    # ── 4) Lưu CSV/JSON ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    cols = ['method', 'tag', 'crf_roi', 'crf_non', 'bitrate_kbps', 'file_size_kb',
            'psnr', 'ssim', 'sa_psnr', 'sa_ssim', 'miou', 'iiou']
    df = df[[c for c in cols if c in df.columns]]
    df.sort_values(['method', 'bitrate_kbps'], inplace=True)

    print("\n" + "=" * 80)
    print("=== SUMMARY (sorted by method, bitrate) ===")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    csv_path  = out_dir / 'rd_curve.csv'
    json_path = out_dir / 'rd_curve.json'
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps({'config': vars(args), 'rows': rows}, indent=2, ensure_ascii=False))

    # ── 5) Plot ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = {
        'h264':  dict(color='#1f77b4', marker='o', label='H.264 baseline'),
        'h265':  dict(color='#2ca02c', marker='s', label='H.265 baseline'),
        'sa264': dict(color='#d62728', marker='^', label='SA-X264'),
        'sa265': dict(color='#9467bd', marker='D', label='SA-X265'),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        sub = df[df['method'] == method].sort_values('bitrate_kbps')
        if sub.empty or sub['miou'].isna().all():
            continue
        ax.plot(sub['bitrate_kbps'], sub['miou'] * 100,
                linewidth=2, markersize=8, **style[method])
        for _, row in sub.iterrows():
            label = (f"CRF {row['crf_roi']}" if row['crf_roi'] == row['crf_non']
                     else f"({row['crf_roi']},{row['crf_non']})")
            ax.annotate(label, (row['bitrate_kbps'], row['miou'] * 100),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=8, color=style[method]['color'])

    ax.set_xlabel('Bitrate (kbps)')
    ax.set_ylabel('mIoU (%)')
    ax.set_title(f"RD Curve — bitrate vs mIoU "
                 f"({args.num_frames} frames, {args.split})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()

    png_path = out_dir / 'rd_curve_bitrate_vs_miou.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")
    print(f"Saved → {png_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
