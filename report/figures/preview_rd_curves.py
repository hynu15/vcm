"""Render preview PNG của 2 RD curve dùng trong ket_qua_thuc_nghiem.tex.

Số liệu khớp 1-1 với pgfplots trong file LaTeX.
"""
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# -- H.264 group ---------------------------------------------------------------
h264 = {
    'bitrate': [5.5, 8.6, 13.0, 20.0, 30.0],
    'miou':    [82.0, 85.5, 88.4, 91.0, 92.6],
}
sac_ccnet_264 = {
    'bitrate': [4.7, 7.5, 11.5, 17.5, 26.5],
    'miou':    [86.6, 89.0, 91.0, 92.5, 93.6],
}
sac_pidnet_264 = {
    'bitrate': [4.9, 7.7, 11.8, 17.8, 26.7],
    'miou':    [85.1, 87.8, 90.0, 91.7, 93.0],
}

# -- H.265 group ---------------------------------------------------------------
h265 = {
    'bitrate': [3.0, 4.8, 7.5, 12.0, 18.0],
    'miou':    [80.8, 84.5, 87.6, 90.3, 92.0],
}
sac_ccnet_265 = {
    'bitrate': [2.8, 4.4, 6.8, 10.5, 16.0],
    'miou':    [85.2, 87.9, 90.1, 91.9, 93.1],
}
sac_pidnet_265 = {
    'bitrate': [2.9, 4.5, 6.9, 10.7, 16.2],
    'miou':    [83.8, 86.7, 89.0, 91.1, 92.5],
}


def render(group_name, baseline, baseline_label, sac_cc, sac_pd, xmax, ymin):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(baseline['bitrate'], baseline['miou'],
            color='#1f77b4', marker='s', markersize=7,
            linewidth=2, label=baseline_label)
    ax.plot(sac_cc['bitrate'], sac_cc['miou'],
            color='#d62728', marker='o', markersize=7,
            linewidth=2, label=f'SAC-CCNet ({group_name})')
    ax.plot(sac_pd['bitrate'], sac_pd['miou'],
            color='#2ca02c', marker='^', markersize=8,
            linewidth=2, label=f'SAC-PIDNet ({group_name})')
    ax.set_xlabel('Bitrate (Mbps)')
    ax.set_ylabel('mIoU (%)')
    ax.set_title(f'{baseline_label} vs SAC-CCNet ({group_name}) vs '
                 f'SAC-PIDNet ({group_name})')
    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=ymin)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


fig1 = render('X264', h264, 'H.264', sac_ccnet_264, sac_pidnet_264, 33, 80)
fig1.savefig(HERE / 'rd_curve_h264.png', dpi=150)

fig2 = render('X265', h265, 'H.265', sac_ccnet_265, sac_pidnet_265, 20, 78)
fig2.savefig(HERE / 'rd_curve_h265.png', dpi=150)


# -- FPS bar chart -------------------------------------------------------------
import numpy as np

configs   = ['CPU\n(Ryzen 9 5900HX)',
             'RTX 4090\n+ libx265',
             'RTX 4090\n+ NVENC',
             'RTX 4090 + NVENC\n+ pipeline']
fps_ccnet  = [0.78,  6.4, 11.7, 17.2]
fps_pidnet = [1.59,  6.5, 13.0, 29.7]

x = np.arange(len(configs))
width = 0.38

fig3, ax = plt.subplots(figsize=(10.5, 5.5))
b1 = ax.bar(x - width/2, fps_ccnet,  width,
            color='#1f77b4', edgecolor='#0d3b66', label='SAC-CCNet')
b2 = ax.bar(x + width/2, fps_pidnet, width,
            color='#2ca02c', edgecolor='#0a4f08', label='SAC-PIDNet')

ax.axhline(30, color='#d62728', linestyle='--', linewidth=1.5)
ax.text(3.5, 30.7, '30 FPS (mục tiêu realtime)',
        ha='right', color='#d62728', fontsize=10, fontweight='bold')

for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.5, f'{h:.1f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('FPS')
ax.set_title('Thông lượng SAC trên CPU vs RTX 4090 (lý thuyết)')
ax.set_ylim(0, 35)
ax.grid(True, axis='y', alpha=0.3)
ax.legend(loc='upper left')
fig3.tight_layout()
fig3.savefig(HERE / 'fps_comparison.png', dpi=150)


# -- Stage breakdown bar (stacked) --------------------------------------------
stages = ['Seg total', 'Stream sep', 'Encode parallel', 'Decode + Merge']
ccnet_cpu    = [866.1,   33.7, 315.0,  68.7]
pidnet_cpu   = [143.3,   65.7, 340.3,  78.8]
ccnet_4090   = [58.0,    16.8,  63.0,  17.5]
pidnet_4090  = [33.7,    32.8,  68.0,  19.8]
ccnet_nvenc  = [58.0,    16.8,   7.0,   3.5]
pidnet_nvenc = [33.7,    32.8,   7.0,   3.5]

labels = ['CCNet\nCPU', 'PIDNet\nCPU',
          'CCNet\n4090+libx265', 'PIDNet\n4090+libx265',
          'CCNet\n4090+NVENC', 'PIDNet\n4090+NVENC']
data   = np.array([ccnet_cpu, pidnet_cpu, ccnet_4090, pidnet_4090,
                   ccnet_nvenc, pidnet_nvenc])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig4, ax = plt.subplots(figsize=(11, 5.5))
bottom = np.zeros(len(labels))
for i, s in enumerate(stages):
    ax.bar(labels, data[:, i], bottom=bottom, color=colors[i],
           edgecolor='white', linewidth=0.6, label=s)
    bottom += data[:, i]

totals = data.sum(axis=1)
for j, t in enumerate(totals):
    ax.text(j, t + 20, f'{t:.0f} ms\n({1000/t:.1f} FPS)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('ms / frame')
ax.set_title('Phân bố thời gian các stage của pipeline SAC')
ax.set_ylim(0, 1500)
ax.grid(True, axis='y', alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
fig4.tight_layout()
fig4.savefig(HERE / 'stage_breakdown.png', dpi=150)


print(f"Saved → {HERE / 'rd_curve_h264.png'}")
print(f"Saved → {HERE / 'rd_curve_h265.png'}")
print(f"Saved → {HERE / 'fps_comparison.png'}")
print(f"Saved → {HERE / 'stage_breakdown.png'}")
