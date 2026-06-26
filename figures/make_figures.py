#!/usr/bin/env python3
"""Generate figures for the SUID safe-sleep confounders literature review.
All numbers are anchored to documented values from cited sources (see review text).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

CB = {  # colorblind-friendly palette (Okabe-Ito)
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
    "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
}

# ---------------------------------------------------------------------------
# FIGURE 1: The diagnostic shift — what actually happened to the death codes
# ---------------------------------------------------------------------------
def fig1_diagnostic_shift():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: SUID two-era decline (anchor points, Erck Lambert 2018; Guare 2024)
    eras_x = [1990, 1998, 2015, 2020, 2021]
    suid =   [154.6, 85.6, 92.4, 96.4, 102.7]  # documented anchor values
    ax1.plot(eras_x, suid, "-o", color=CB["blue"], lw=2.4, ms=7, zorder=3)
    ax1.axvspan(1990, 1998, color=CB["green"], alpha=0.12)
    ax1.axvspan(1999, 2015, color=CB["orange"], alpha=0.12)
    ax1.axvspan(2019.5, 2021.5, color=CB["red"], alpha=0.12)
    ax1.text(1994, 158, "Genuine decline\n-44.6% (1990-98)", ha="center",
             fontsize=8.5, color=CB["green"], fontweight="bold")
    ax1.text(2007, 118, "Stall\n-7% (1999-2015)", ha="center",
             fontsize=8.5, color=CB["orange"], fontweight="bold")
    ax1.text(2020.3, 88, "Rise\n(post-2020)", ha="center",
             fontsize=8.5, color=CB["red"], fontweight="bold")
    ax1.set_title("A. SUID rate: one real decline, then a stall",
                  fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("SUID deaths per 100,000 live births", fontsize=9)
    ax1.set_xlabel("Year", fontsize=9)
    ax1.set_ylim(60, 170)
    ax1.grid(True, alpha=0.25)

    # Panel B: composition change 1999-2015 (Erck Lambert 2018)
    cats = ["SIDS\n(R95)", "ASSB\n(W75)", "Unknown\n(R99)"]
    pct = [-35.8, 183.8, 0.0]
    colors = [CB["blue"], CB["red"], CB["grey"]]
    bars = ax2.bar(cats, pct, color=colors, edgecolor="black", lw=0.6)
    ax2.axhline(0, color="black", lw=0.8)
    for b, v in zip(bars, pct):
        off = 6 if v >= 0 else -16
        ax2.text(b.get_x()+b.get_width()/2, v+off,
                 f"{v:+.1f}%" if v != 0 else "~0%(NS)",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("B. Same period (1999-2015): deaths moved between codes",
                  fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Change in cause-specific rate", fontsize=9)
    ax2.set_ylim(-70, 215)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Figure 1. The diagnostic-shift baseline: anchor on SUID, not SIDS",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig1_diagnostic_shift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 2: Causal-chain DAG with evidence strength
# ---------------------------------------------------------------------------
def fig2_causal_dag():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, text, fc, tc="black", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                     fc=fc, ec="black", lw=1.0, alpha=0.95))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, color, label="", lw=2.2, style="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=18, color=color, lw=lw,
                     linestyle=style, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2+0.18, label, ha="center", fontsize=7.5,
                    color=color, style="italic")

    # exposure
    box(0.2, 3.4, 1.9, 1.2, "Safe-sleep\nguidelines\n(1994+)", CB["sky"])
    # mediators
    box(3.0, 5.6, 2.1, 1.0, "Parental sleep\ndeprivation /\nfragmentation", "#FDE9C8")
    box(3.0, 3.4, 2.1, 1.0, "Deteriorated\nparental\nmental health", "#FDE9C8")
    box(3.0, 1.2, 2.1, 1.0, "Guideline-\ndiscordant\nbedsharing", "#FDE9C8")
    # outcomes
    box(6.6, 6.2, 2.3, 1.0, "Thread 2:\nAbusive head trauma", "#F4CCCC")
    box(6.6, 4.6, 2.3, 1.0, "Thread 1:\n(maternal MH outcome)", "#F4CCCC")
    box(6.6, 3.0, 2.3, 1.0, "Thread 3:\nMaternal suicide", "#F4CCCC")
    box(6.6, 1.2, 2.3, 1.0, "SUID\n(overlay when impaired)", "#D9EAD3")

    # arrows w/ evidence strength color: green strong / orange moderate / red weak-untested
    arrow(2.1, 4.2, 3.0, 5.9, CB["orange"], "weak (Link A)")          # guideline->sleep loss
    arrow(2.1, 4.0, 3.0, 3.9, CB["red"], "weak/indirect")             # guideline->MH
    arrow(2.1, 3.7, 3.0, 1.7, CB["orange"], "")                       # guideline->bedshare
    arrow(5.1, 6.0, 6.6, 6.7, CB["green"], "strong (Link C)")         # sleeploss->AHT (crying/stress)
    arrow(5.1, 5.9, 6.6, 5.1, CB["orange"], "")                       # sleeploss->MH
    arrow(5.1, 3.9, 6.6, 5.0, CB["red"], "")                          # MH->MH outcome
    arrow(5.1, 3.7, 6.6, 3.5, CB["red"], "weak link")                 # MH->suicide
    arrow(5.1, 1.7, 6.6, 1.7, CB["green"], "established")             # bedshare->SUID

    # trend counter-evidence flags
    ax.text(7.75, 5.75, "TREND: declined\n2009-2014", ha="center", fontsize=7,
            color=CB["red"], fontweight="bold")
    ax.text(7.75, 2.55, "TREND: counter-\ndirectional", ha="center", fontsize=7,
            color=CB["red"], fontweight="bold")

    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], color=CB["green"], lw=3, label="Well-supported link"),
           Line2D([0],[0], color=CB["orange"], lw=3, label="Partial / indirect link"),
           Line2D([0],[0], color=CB["red"], lw=3, label="Weak / untested link")]
    ax.legend(handles=leg, loc="lower left", fontsize=8.5, frameon=True,
              bbox_to_anchor=(0.0, -0.02))

    ax.set_title("Figure 2. Hypothesized causal chains and the strength of evidence for each link",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig("figures/fig2_causal_dag.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 3: Hypothesis scorecard
# ---------------------------------------------------------------------------
def fig3_scorecard():
    threads = ["T1 Parental\nmental health", "T2 Abusive\nhead trauma",
               "T3 Maternal\nsuicide", "T4 Sedative\nprescribing"]
    dims = ["Mechanistic\nplausibility", "Direct test\nexists?",
            "Trend\nconsistency", "Link to\nsafe-sleep"]
    # 0=weak/no(red) 1=mixed(orange) 2=strong/yes(green)
    M = np.array([
        [2, 0, 0, 2],   # T1
        [2, 0, 0, 1],   # T2
        [1, 0, 1, 0],   # T3
        [1, 0, 0, 1],   # T4
    ])
    cmap = matplotlib.colors.ListedColormap([CB["red"], CB["orange"], CB["green"]])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    labels = {0: "weak / no", 1: "mixed", 2: "strong / yes"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, labels[M[i, j]], ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, fontsize=9)
    ax.set_yticks(range(len(threads))); ax.set_yticklabels(threads, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(dims), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(threads), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Figure 3. Hypothesis scorecard: where each thread stands today",
                 fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig("figures/fig3_scorecard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIGURE 4: PRISMA-style flow (narrative/rapid review)
# ---------------------------------------------------------------------------
def fig4_prisma():
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    def box(y, text, fc=CB["sky"], h=1.2, x=1.2, w=7.6, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
                     fc=fc, ec="black", lw=1.0, alpha=0.9))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)

    def varrow(y1, y2, x=5.0):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                     mutation_scale=16, color="black", lw=1.4))

    box(10.5, "IDENTIFICATION\n6 parallel topic searches (WebSearch / PubMed,\nPMC, CDC, Semantic Scholar, journal sites)",
        "#E8E8E8", h=1.3)
    varrow(10.5, 9.8)
    box(8.6, "Records surfaced across 6 streams\n(~45-70 candidate sources screened on title/abstract)",
        CB["sky"], h=1.2)
    varrow(8.6, 7.9)
    box(6.7, "SCREENING\nExcluded: non-peer-reviewed, off-topic,\nunverifiable, redundant guideline restatements",
        "#FDE9C8", h=1.3)
    varrow(6.7, 6.0)
    box(4.8, "ELIGIBILITY\nFull records checked; DOI/PMID verified;\ndirectional findings extracted",
        "#FDE9C8", h=1.3)
    varrow(4.8, 4.1)
    box(2.9, "INCLUDED: ~60 verified sources\nBaseline 11 | T1 11 | T2 13 | T3 11 | T4 13 | Methods 14",
        CB["green"], h=1.3, fs=9)
    ax.text(5.0, 1.6, "(some sources serve multiple threads)", ha="center",
            fontsize=8, style="italic", color=CB["grey"])

    ax.set_title("Figure 4. Source identification and screening flow",
                 fontsize=11.5, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig("figures/fig4_prisma.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_diagnostic_shift()
    fig2_causal_dag()
    fig3_scorecard()
    fig4_prisma()
    print("All figures written to figures/")
