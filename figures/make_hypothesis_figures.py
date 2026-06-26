#!/usr/bin/env python3
"""Figures for the Narrative-1 hypothesis-generation report."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

CB = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00",
      "purple":"#CC79A7","grey":"#999999","sky":"#56B4E9","yellow":"#F0E442"}

# ---------- Figure 1: competing mechanisms ----------
def fig1():
    fig, ax = plt.subplots(figsize=(12, 6.6)); ax.set_xlim(0,12); ax.set_ylim(0,9); ax.axis("off")
    def box(x,y,w,h,t,fc,fs=8.5,tc="black"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.1",
                     fc=fc,ec="black",lw=1.0,alpha=0.95))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold")
    def arr(x1,y1,x2,y2,c,lw=2.0,style="-"):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,
                     color=c,lw=lw,linestyle=style,shrinkA=3,shrinkB=3))
    # exposure
    box(0.2,3.9,2.0,1.4,"Safe-sleep\nguideline regime\n(messaging + norms)",CB["sky"],8.5)
    # four mechanism boxes
    box(3.4,6.9,3.1,1.0,"H1 threat-vigilance →\nsafe-sleep intrusive thoughts",  "#D6EAF8")
    box(3.4,5.3,3.1,1.0,"H2 unattainable norm + deviation\n→ guilt / low self-efficacy","#D5F5E3")
    box(3.4,3.7,3.1,1.0,"H3 adherence →\nfragmented parental sleep","#FCF3CF")
    box(3.4,1.9,3.1,1.1,"H5 (null) adversity confounds\n+ screening inflates measure","#FADBD8")
    # outcome
    box(8.2,4.0,3.0,1.3,"Parental depression /\nanxiety, low competence\n(mothers & fathers)","#E8DAEF",8.5)
    # arrows exposure->mechanisms
    arr(2.2,4.9,3.4,7.4,CB["blue"]); arr(2.2,4.8,3.4,5.8,CB["green"])
    arr(2.2,4.6,3.4,4.2,CB["orange"]);
    arr(2.2,4.3,3.4,2.5,CB["red"],style="--")  # null/confound dashed
    # mechanisms->outcome
    arr(6.5,7.4,8.2,5.1,CB["blue"]); arr(6.5,5.8,8.2,4.9,CB["green"])
    arr(6.5,4.2,8.2,4.6,CB["orange"]); arr(6.5,2.4,8.2,4.2,CB["red"],style="--")
    # H4 SES modifier band
    ax.add_patch(FancyBboxPatch((3.3,0.5),3.3,0.8,boxstyle="round,pad=0.04,rounding_size=0.08",
                 fc="#F9E79F",ec=CB["orange"],lw=1.4,alpha=0.9))
    ax.text(4.95,0.9,"H4  Socioeconomic adversity MODIFIES all pathways\n(amplifies in deprived / surveilled families)",
            ha="center",va="center",fontsize=8,style="italic",color="#7D6608",fontweight="bold")
    for mx in (6.6,):  # modifier arrows up
        for ty in (4.2,5.8,7.4,2.4):
            arr(6.6,1.3,6.7,ty-0.4,CB["orange"],lw=1.0,style=":")
    leg=[Line2D([0],[0],color=CB["blue"],lw=3,label="H1 cognitive vigilance"),
         Line2D([0],[0],color=CB["green"],lw=3,label="H2 guilt / self-efficacy"),
         Line2D([0],[0],color=CB["orange"],lw=3,label="H3 sleep deprivation"),
         Line2D([0],[0],color=CB["red"],lw=3,ls="--",label="H5 confound / ascertainment (null)"),
         Line2D([0],[0],color=CB["orange"],lw=1.5,ls=":",label="H4 SES moderation")]
    ax.legend(handles=leg,loc="upper center",ncol=3,fontsize=8,frameon=True,bbox_to_anchor=(0.5,1.06))
    ax.set_title("Figure 1. Competing mechanisms linking the safe-sleep regime to parental mental health",
                 fontsize=12,fontweight="bold",pad=24)
    fig.tight_layout(); fig.savefig("figures/hyp_fig1_mechanisms.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# ---------- Figure 2: discriminating-predictions matrix ----------
def fig2():
    rows=["H1 cognitive\nvigilance","H2 guilt /\nself-efficacy","H3 sleep\ndeprivation","H5 confound /\nascertainment (null)"]
    cols=["Distress tracks\nADHERENCE","Distress tracks\nthreat-cognition / ITs","Mediated by\nguilt / low efficacy",
          "Larger in\nlow-SES (H4)","Survives adversity ctrl\n+ ascertainment-robust"]
    # 0=No(red) 1=partial/mixed(amber) 2=Yes(green)
    M=np.array([
        [0,2,0,1,2],   # H1
        [1,0,2,2,2],   # H2
        [2,0,0,1,2],   # H3
        [0,0,0,1,0],   # H5 (diagnostic: does NOT survive)
    ])
    txt={0:"No",1:"mixed",2:"Yes"}
    cmap=matplotlib.colors.ListedColormap([CB["red"],CB["orange"],CB["green"]])
    fig,ax=plt.subplots(figsize=(11,4.6))
    ax.imshow(M,cmap=cmap,vmin=0,vmax=2,aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j,i,txt[M[i,j]],ha="center",va="center",fontsize=9,fontweight="bold",color="white")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,fontsize=8.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=9)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True); ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="minor",length=0)
    ax.set_title("Figure 2. Discriminating predictions — how to tell the hypotheses apart",
                 fontsize=11.5,fontweight="bold",pad=10)
    fig.text(0.5,-0.04,"Key discriminator: H1/H2 predict distress is HIGHER in non-adherent parents (who deviate to cope); "
             "H3 predicts the opposite. H5 is the only account that does NOT survive adversity control + ascertainment-robust design.",
             ha="center",fontsize=8,style="italic",wrap=True)
    fig.tight_layout(); fig.savefig("figures/hyp_fig2_discriminating.png",dpi=200,bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    fig1(); fig2(); print("wrote figures/hyp_fig1_mechanisms.png, figures/hyp_fig2_discriminating.png")
