"""개인 보고서용 그림 두 장: 1차 영상 판정 타임라인, 판정 파이프라인."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

S = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Pretendard", "axes.unicode_minus": False, "font.size": 10})
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(p):
    rows = list(csv.DictReader(open(p)))
    return [float(r["t"]) / 60 for r in rows], [float(r["in_zone_conf"]) for r in rows], [r["state"] for r in rows]


# 그림 1. 1차 영상 판정 타임라인
t26, c26, s26 = load(S / "log26.csv")
tv2, cv2, sv2 = load(S / "verify_v2/log.csv")
end = tv2[-1]
cut = 0.8 * end

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 4.6), gridspec_kw={"height_ratios": [4, 1.1], "hspace": 0.12}, sharex=True)
fig.patch.set_facecolor(SURF)
for a in (ax, ax2):
    a.set_facecolor(SURF)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.spines["left"].set_color(MUTED)
    a.spines["bottom"].set_color(MUTED)
    a.tick_params(colors=INK2)

for x0, x1 in [(0, 210 / 60), (1148 / 60, end)]:
    ax.axvspan(x0, x1, color="#efeeea", lw=0, zorder=0)
ax.text(1.75, 1.04, "정답: 사람 있음", ha="center", color=INK2, fontsize=9)
ax.text((1148 / 60 + end) / 2, 1.04, "정답: 사람 있음", ha="center", color=INK2, fontsize=9)
ax.text(11.3, 1.04, "정답: 짐만 남음 (3:32~19:06)", ha="center", color=INK2, fontsize=9)

ax.plot(t26, c26, color=BLUE, lw=1.6, label="사전학습 YOLO26s", zorder=3)
ax.plot(tv2, cv2, color=ORANGE, lw=1.6, label="파인튜닝 YOLO26s v2", zorder=3)
for th, lab in [(0.30, "26s 기준 0.30"), (0.35, "v2 기준 0.35")]:
    ax.axhline(th, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
ax.text(8.0, 0.255, "기준 0.30 / 0.35", color=INK2, fontsize=8.5)
ax.axvline(cut, color=INK2, lw=1, zorder=2)
ax.text(cut - 0.15, 0.13, "17:34 이후는 v2 학습에 쓰지 않은 구간 →", ha="right", color=INK, fontsize=8.5)
ax.annotate("3:30 사람이 나가는 순간\n26s 0.29로 놓침", xy=(3.5, 0.289), xytext=(4.6, 0.55), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_ylim(-0.03, 1.1)
ax.set_ylabel("감지 영역 안 사람 신뢰도", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(loc="center left", bbox_to_anchor=(0.36, 0.42), frameon=False, fontsize=9)

# 아래 띠: 판정된 신호등 색
for row, (tt, ss) in enumerate([(tv2, sv2), (t26, s26)]):
    for x, st in zip(tt, ss):
        ax2.add_patch(plt.Rectangle((x, row + 0.15), 2 / 60, 0.7, color="#e34948" if st == "red" else "#eda100", lw=0))
ax2.set_yticks([0.5, 1.5], ["v2", "26s"])
ax2.set_ylim(0, 2)
ax2.set_xlim(0, end + 2 / 60)
ax2.set_xlabel("영상 시각 (분)", color=INK2)
ax2.text(end + 0.4, 1.0, "빨강 사용중\n노랑 비움", fontsize=8, color=INK2, va="center")
fig.savefig(OUT / "fig_timeline.png", dpi=220, bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# 그림 2. 판정 파이프라인
fig, ax = plt.subplots(figsize=(9, 2.9))
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.set_xlim(0, 100)
ax.set_ylim(0, 32)
ax.axis("off")


def box(x, y, w, h, title, body, dark=False, tdy=2.6, bdy=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#1d3a3a" if dark else "#ffffff", ec="#1d3a3a" if dark else MUTED, lw=1))
    ax.text(x + w / 2, y + h - tdy, title, ha="center", va="top", fontsize=9.5, weight="bold", color="#ffffff" if dark else INK)
    ax.text(x + w / 2, y + h - bdy, body, ha="center", va="top", fontsize=8, color="#dfe9e7" if dark else INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))


box(1, 7, 15, 20, "프레임 수집", "capture.frames()\n영상: 영상 시각\n스트림: 실제 시각\n끊기면 (t, None)")
box(21, 7, 15, 20, "사람 검출", "YOLO26s\n입력 960\nperson 클래스만")
box(41, 7, 15, 20, "감지 영역", "박스 중심이\n다각형 안이면\n영역 안 사람")
box(61, 4, 17, 26, "상태 판정", "2프레임 연속 → 빨강\n비움 30분 미만 → 노랑\n비움 30분 이상 → 초록\n끊긴 시간은\n비움에 넣지 않음", dark=True)
for y, t, b in [(23, "신호등", "색이 바뀌면 POST 전송"), (12.5, "스냅샷", "사람 모자이크, 1일 보관"), (2, "판정 기록", "CSV, 줄마다 바로 기록")]:
    box(83, y, 16, 8.5, t, b, tdy=1.6, bdy=5.0)
arrow(16.8, 17, 20.2, 17)
arrow(36.8, 17, 40.2, 17)
arrow(56.8, 17, 60.2, 17)
for y in (27.25, 16.75, 6.25):
    arrow(78.8, 17, 82.2, y)
fig.savefig(OUT / "fig_pipeline.png", dpi=220, bbox_inches="tight", facecolor=SURF)
plt.close(fig)
print("ok")
