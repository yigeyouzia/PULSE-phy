#!/usr/bin/env python3
"""Render the implemented PULSE_FACT path and labelled research opportunities.

Single-canvas source-grounded schematic, with no simulated experimental data.
PNG is the requested planning figure; SVG/PDF retain editable text for revision.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


PULSE_FILL, PULSE_EDGE = "#EDF2FA", "#53729D"
FACT_FILL, FACT_EDGE = "#E3F3EF", "#347F73"
IDEA_FILL, IDEA_EDGE = "#FFF4E5", "#AD6518"
INK, MUTED = "#233247", "#596879"


def render(output_dir):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "axes.linewidth": 0.8})
    # Large planning canvas. A submission version needs re-layout at venue width.
    fig = plt.figure(figsize=(18.8, 11.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 20), ylim=(0, 12))
    ax.axis("off")

    def text(x, y, value, size=11, weight="normal", color=INK, ha="center", va="center"):
        return ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
                       ha=ha, va=va, linespacing=1.4)

    def tag(x, y, label):
        ax.add_patch(Circle((x, y), 0.17, facecolor=IDEA_FILL, edgecolor=IDEA_EDGE, lw=1.0, zorder=5))
        text(x, y, label, size=9, weight="bold", color=IDEA_EDGE).set_zorder(6)

    def box(x, y, w, h, title, detail, family="pulse", marker=None):
        fill, edge = ((FACT_FILL, FACT_EDGE) if family == "fact" else
                      (IDEA_FILL, IDEA_EDGE) if family == "idea" else (PULSE_FILL, PULSE_EDGE))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2, edgecolor=edge, facecolor=fill, linestyle="--" if family == "idea" else "-"))
        text(x + w / 2, y + h - 0.30, title, size=11.5, weight="bold")
        text(x + w / 2, y + h / 2 - 0.20, detail, size=10.3)
        if marker:
            tag(x + 0.22, y + h + 0.17, marker)

    def arrow(points, candidate=False, width=1.15):
        color = IDEA_EDGE if candidate else MUTED
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=width,
                    linestyle=(0, (4, 3)) if candidate else "-", zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=10,
            color=color, lw=width, linestyle=(0, (4, 3)) if candidate else "-", zorder=1))

    text(0.45, 11.60, "PULSE_FACT: implemented model and candidate innovation sites", 18, "bold", ha="left")
    text(0.45, 11.15, "Source-grounded planning diagram  |  solid arrows: current data flow  |  orange labels: research opportunities", 11, color=MUTED, ha="left")
    for x, fill, edge, label in [(0.5, PULSE_FILL, PULSE_EDGE, "Inherited PULSE components"),
                                (6.0, FACT_FILL, FACT_EDGE, "Adapted FACT interaction"),
                                (11.0, IDEA_FILL, IDEA_EDGE, "Candidate only; not implemented")]:
        ax.add_patch(FancyBboxPatch((x, 10.57), 0.36, 0.24, boxstyle="round,pad=0.01",
            facecolor=fill, edgecolor=edge, lw=1))
        text(x + 0.5, 10.69, label, size=10.5, ha="left")

    # Phase reference is reused in normalisation and in future reference routing.
    box(2.55, 8.65, 3.8, 1.25, "Historical phase reference", "Codebook + TimeEnc_x\nA_x: [B,L,C]", marker="I1")
    text(0.50, 9.65, "Cycle index\nHistorical timestamps", 10.5, ha="left")
    arrow([(1.92, 9.12), (2.55, 9.12)])
    box(12.55, 8.65, 4.8, 1.25, "Future reference / Phase Router", "A_y = Router(Y_0, A_x) + T_y\nPatch-wise routing; actual attention heads = 1", marker="I7")
    box(17.90, 8.65, 1.55, 1.25, "Known future", "TimeEnc_y\nT_y: [B,H,C]", family="pulse")
    arrow([(6.35, 9.28), (12.55, 9.28)])
    text(9.40, 9.55, "A_x (not mixed by SAM)", 10.3, color=MUTED)
    arrow([(17.90, 9.28), (17.35, 9.28)])

    y, h = 5.30, 1.35
    box(0.45, y, 1.50, h, "History X", "[B,L,C]", family="pulse")
    box(2.55, y, 2.15, h, "Reference norm", "RevIN(X - A_x)\n+ A_x", marker="I2")
    box(5.25, y, 1.75, h, "SAM", "Train only\nMix input + stats", marker="I3")
    box(7.55, y, 1.65, h, "Temporal MLP", "Linear(L,512)\nGELU", family="pulse")
    box(9.75, y, 2.25, h, "FACT block", "[B,C,512]\nOne block: N = 1", family="fact", marker="I4")
    box(12.55, y, 1.60, h, "Output map", "Linear(512,H)\nY_0: [B,H,C]", family="pulse")
    box(14.70, y, 2.65, h, "Paired restoration", "RevIN inverse(Y_0 - A_y)\n+ A_y", family="pulse")
    box(17.90, y, 1.55, h, "Forecast", "Y_hat\n[B,H,C]", family="pulse")
    spans = [(0.45, 1.50), (2.55, 2.15), (5.25, 1.75), (7.55, 1.65),
             (9.75, 2.25), (12.55, 1.60), (14.70, 2.65), (17.90, 1.55)]
    for (x0, w0), (x1, _) in zip(spans[:-1], spans[1:]):
        arrow([(x0 + w0, y + h / 2), (x1, y + h / 2)])
    arrow([(3.60, 8.65), (3.60, 6.65)])
    arrow([(13.35, 6.65), (13.35, 8.65)])
    text(13.68, 7.52, "Y_0", 10.3, color=MUTED)
    arrow([(16.05, 8.65), (16.05, 6.65)])
    text(16.40, 7.52, "A_y", 10.3, color=MUTED)

    # Stats are mixed at SAM, then reused for inversion. This bypass is current.
    arrow([(6.10, 5.30), (6.10, 4.74), (15.72, 4.74), (15.72, 5.30)])
    text(10.8, 4.98, "Paired mean / variance (mixed in training)", 10.0, color=MUTED)

    # The dashed condition edge is explicitly a candidate, not an existing link.
    arrow([(5.35, 8.65), (5.35, 7.90), (10.85, 7.90), (10.85, 6.65)], candidate=True)
    text(8.1, 8.16, "Candidate I4: phase-conditioned interaction", 10.5, color=IDEA_EDGE)
    text(8.1, 7.59, "Current FACT block has no direct phase-conditioning input", 10, color=MUTED)

    # A zoom of the actual single block: branches are parallel, not a depth stack.
    ax.add_patch(FancyBboxPatch((0.45, 0.92), 12.60, 3.28, boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="white", edgecolor=FACT_EDGE, linewidth=1.2))
    text(0.78, 3.85, "Inside the current FACT block (default settings)", 12, "bold", color=FACT_EDGE, ha="left")
    box(0.78, 2.12, 2.20, 1.15, "Grid adapter", "Cyclic pad; reshape\n[B,512,g,g]", family="fact")
    box(3.42, 2.12, 2.70, 1.15, "Parallel DW branches", "BN + kernels {1,3}\nDilations {1,2}; sum + U", family="fact")
    box(6.57, 2.12, 2.83, 1.15, "Pointwise FFN", "BN; 512 > 256 > 512\nGELU; dropout 0.1; residual", family="fact")
    box(9.85, 2.12, 2.82, 1.15, "Crop + transpose", "g = floor(sqrt(C)) + 1\nBack to [B,C,512]", family="fact")
    for start, end in [(2.98, 3.42), (6.12, 6.57), (9.40, 9.85)]:
        arrow([(start, 2.69), (end, 2.69)])
    tag(0.98, 1.63, "I5")
    text(1.30, 1.63, "Candidate: stack / alternate blocks; N=2 or 3 is not implemented", 10.3, ha="left", color=IDEA_EDGE)
    tag(0.98, 1.20, "I6")
    text(1.30, 1.20, "Candidate: selective residual or time / frequency branches (not the current model)", 10.3, ha="left", color=IDEA_EDGE)

    box(13.60, 2.48, 5.85, 1.72, "Training objective (outside Model.forward)",
        "lambda_rec * MSE + lambda_fft * mean |FFT(error)|\nCurrent scripts: (lambda_rec, lambda_fft) = (0, 1)\nTarget uses the same SAM lambda and permutation", family="pulse", marker="I8")
    arrow([(18.68, 5.30), (18.68, 4.20)])
    box(13.60, 0.92, 5.85, 1.20, "Candidate losses / physical consistency", "Time + frequency balance; differences; phase-aware terms\nPhysical constraints require valid variables, units and laws", family="idea")

    text(0.45, 0.43,
        "B=batch; L=history; H=horizon; C=variables. SAM bypassed at inference; A_x and timestamps remain unmixed. Grid is not a physical topology.",
        10, color=MUTED, ha="left")
    fig.canvas.draw()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "pulse_fact_architecture_innovation"
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        # This is one schematic canvas, with no comparable plot-area panels.
        (stem.with_suffix(".alignment.json")).write_text(json.dumps({"status": "NOT APPLICABLE",
            "reason": "Single schematic axes; no comparable statistical panels",
            "width_pt": 18.8 * 72, "height_pt": 11.2 * 72}, indent=2) + "\n")
    else:
        require_matplotlib_panel_alignment(fig, json_out=str(stem.with_suffix(".alignment.json")),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(stem) + ".png", dpi=300)
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".pdf")
    plt.close(fig)
    print("Saved PNG / SVG / PDF:", stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    render(parser.parse_args().output_dir)
