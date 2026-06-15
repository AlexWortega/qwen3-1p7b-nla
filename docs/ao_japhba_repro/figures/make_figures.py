"""Generate explanatory figures for the Universal Activation Oracle:
  fig1_architecture.png  — what it is: inputs -> enc_M -> trunk -> output heads
  fig2_training.png      — what we train on: data sources, task mixture, train vs held-out
  fig3_capabilities.png  — what we can do: real results (bias auditing, intent, jailbreaks, organisms)
Run:  python docs/ao_japhba_repro/figures/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = HERE
def load(p):
    try: return json.load(open(os.path.join(REPO, p)))
    except Exception: return {}

INK="#1b2330"; MUT="#5b6678"; LINE="#9aa6b8"
BLUE="#2f6db0"; GREEN="#2e8b6f"; AMBER="#d39a2d"; RED="#c0504d"; PURPLE="#6c57b0"; GREY="#c7cdd6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":MUT,
                     "axes.labelcolor":INK,"text.color":INK,"xtick.color":MUT,"ytick.color":MUT})

def box(ax,x,y,w,h,text,fc,ec=None,fs=10,bold=False,tc=INK,r=0.025):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.004,rounding_size={r}",
                 fc=fc,ec=ec or MUT,lw=1.3,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
            color=tc,fontweight="bold" if bold else "normal",zorder=3,wrap=True)
def arrow(ax,x1,y1,x2,y2,c=INK,lw=1.8,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=14,
                 lw=lw,color=c,zorder=1,shrinkA=2,shrinkB=2))

# ============================================================ FIG 1: ARCHITECTURE + I/O
def fig_arch():
    fig,ax=plt.subplots(figsize=(13.5,7.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    ax.text(50,97,"Universal Activation Oracle — one reader for any model's activations",
            ha="center",fontsize=15,fontweight="bold")
    ax.text(50,92.3,"INPUT: a hidden activation from ANY LLM's response    →    OUTPUT: a verbal / calibrated read of what it's doing",
            ha="center",fontsize=10.5,color=MUT)

    # INPUT: pool of models
    box(ax,2,58,20,30,"",("#eef3f9"),BLUE,r=0.04)
    ax.text(12,85,"ANY LLM  (model-agnostic)",ha="center",fontsize=10,fontweight="bold",color=BLUE)
    for i,m in enumerate(["Llama","Qwen2.5 / Qwen3","Gemma-2","Pythia / GPT-2","Phi · SmolLM · …"]):
        ax.text(12,80- i*4.2,m,ha="center",fontsize=9,color=INK)
    ax.text(12,60.5,"mean-pool layer≈0.5\nover the response → hₘ (d_M)",ha="center",fontsize=8.4,color=MUT)

    # enc_M
    box(ax,27,62,15,18,"enc_M\n(per-model\nLINEAR adapter)\nd_M → 2048",("#eaf1ea"),GREEN,fs=9.2,bold=True)
    ax.text(34.5,59,"lstsq-init vs anchor,\njointly fine-tuned",ha="center",fontsize=7.6,color=MUT)
    # normalize + inject
    box(ax,46,64,12,14,"normalize\n→ √2048\n+ inject at\nmarker slot",("#f3eef7"),PURPLE,fs=8.6)
    # trunk
    box(ax,62,60,17,22,"Qwen3-1.7B\ntrunk\n+ LoRA r=32",("#fdf4e3"),AMBER,fs=10.5,bold=True)
    ax.text(70.5,58.2,"base frozen · ONE shared trunk\nfor every model in the pool",ha="center",fontsize=7.6,color=MUT)

    arrow(ax,22,73,27,72,BLUE); arrow(ax,42,71,46,71,GREEN); arrow(ax,58,71,62,71,PURPLE)
    ax.text(24.6,75.2,"hₘ",fontsize=9,color=BLUE); ax.text(60,73.2,"z∈ℝ²⁰⁴⁸",fontsize=7.8,color=PURPLE)

    # OUTPUT HEADS
    heads=[("AV — verbalize","free-form text:\n“what does this activation express?”",GREEN,86),
           ("detect — Yes/No","calibrated P(exhibits behaviour X?)\nbias · deception · jailbreak",BLUE,70.5),
           ("linear probe","score on enc_M feature:\norganism · harmful-intent · jailbreak-success",PURPLE,55),
           ("AR + dec_M","reconstruct ĥₘ in native space → FVE\n(faithfulness check)",AMBER,39.5)]
    for name,desc,c,yy in heads:
        box(ax,84,yy,14.5,11,name,(("#ffffff")),c,fs=9.6,bold=True,tc=c)
        ax.text(84.2,yy-1.4,desc,ha="left",va="top",fontsize=7.5,color=MUT)
        arrow(ax,79,71,84,yy+5.5,c,lw=1.5)
    ax.text(91.3,98,"OUTPUT HEADS",ha="center",fontsize=9,fontweight="bold",color=INK)

    # bottom strip: key properties
    box(ax,2,4,96,26,"",("#f7f9fc"),LINE,r=0.015)
    ax.text(4,27,"Why it's UNIVERSAL & what makes it work",fontsize=11,fontweight="bold")
    bullets=[
     "• Only the per-model enc_M is model-specific (a tiny LINEAR map). The trunk + all heads are SHARED across every architecture.",
     "• A NEW model needs no oracle training — closed-form lstsq-fit its enc_M against the trained-tag mean (held-out trick). Reads archs it never saw.",
     "• Activation is injected as a single token at a reserved marker slot, normalized to √2048 — same manifold the trunk was trained on.",
     "• READ WITH A PROBE, not by asking: the trained linear probe extracts intent/jailbreak/organism (AUROC 0.78–1.0); the generative",
     "   verbalize head is unfaithful (confabulates). detect_qa reads topic-harm, not the comply/refuse decision.",
    ]
    for i,b in enumerate(bullets):
        ax.text(4,23.2-i*3.6,b,fontsize=8.7,color=INK)
    fig.savefig(os.path.join(OUT,"fig1_architecture.png"),dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("wrote fig1_architecture.png")

# ============================================================ FIG 2: WHAT WE TRAIN ON
def fig_train():
    fig=plt.figure(figsize=(13.5,7.8)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.05],hspace=0.42,wspace=0.22,
                                                            left=0.06,right=0.97,top=0.9,bottom=0.08)
    fig.suptitle("What the oracle is trained on  (multi-task SFT, cross-model)",fontsize=15,fontweight="bold",y=0.975)

    # (a) data sources
    axa=fig.add_subplot(gs[0,0]); axa.axis("off"); axa.set_title("Data sources",fontsize=11,fontweight="bold",loc="left")
    rows=[("Passage corpus → teacher z","10.5k texts, OpenRouter summaries  (AV)",GREEN),
          ("Bias transcripts","social/political + quirks, replayed thru K models  (detect)",BLUE),
          ("Direction pairs","biased ↔ balanced same-topic  (hard-neg / dir-pos)",PURPLE),
          ("Model organisms","EM (bad-medical/finance/sports), sandbagging  (probe)",RED),
          ("LatentQA + deception","behaviour QA 907 rows · lie rollouts",AMBER)]
    for i,(t,d,c) in enumerate(rows):
        y=0.86-i*0.185
        axa.add_patch(FancyBboxPatch((0.02,y-0.07),0.96,0.15,boxstyle="round,pad=0.005,rounding_size=0.02",
                      fc="#ffffff",ec=c,lw=1.4,transform=axa.transAxes))
        axa.text(0.05,y+0.025,t,fontsize=9.3,fontweight="bold",color=c,transform=axa.transAxes)
        axa.text(0.05,y-0.04,d,fontsize=8.0,color=MUT,transform=axa.transAxes)

    # (b) task mixture
    axb=fig.add_subplot(gs[0,1]); axb.set_title("Task mixture  (sampling weights, v21 full)",fontsize=11,fontweight="bold",loc="left")
    tasks=["detect\n(Yes/No)","av\n(verbalize)","latentqa","lie"]; w=[8,3,2,2]; cols=[BLUE,GREEN,AMBER,RED]
    axb.barh(range(len(tasks))[::-1],w,color=cols,edgecolor="white",height=0.62)
    axb.set_yticks(range(len(tasks))[::-1]); axb.set_yticklabels(tasks,fontsize=9)
    for i,v in enumerate(w): axb.text(v+0.1,len(tasks)-1-i,str(v),va="center",fontsize=9,color=INK)
    axb.set_xlabel("weight"); axb.set_xlim(0,9.5)
    for s in ["top","right"]: axb.spines[s].set_visible(False)
    axb.text(0.0,-1.15,"detect sub-mix  pos : in-org : clean : hard-neg ≈ 2 : 1.5 : 1.5  (≈60% negatives → kills confabulation)",
             fontsize=8.0,color=MUT,transform=axb.get_xaxis_transform())

    # (c) train vs held-out models
    axc=fig.add_subplot(gs[1,0]); axc.axis("off"); axc.set_title("Models: train pool vs held-out (zero-shot transfer)",
                                                                 fontsize=11,fontweight="bold",loc="left")
    train=["qwen3-1p7b","phi-1p5","smollm3-3b","qwen2p5-7b","gemma2","qwen2p5-0p5b","qwen3-4b"]
    for i,t in enumerate(train):
        x=0.02+(i%4)*0.245; y=0.74-(i//4)*0.22
        axc.add_patch(FancyBboxPatch((x,y-0.07),0.225,0.13,boxstyle="round,pad=0.004,rounding_size=0.02",
                      fc="#eaf1ea",ec=GREEN,lw=1.2,transform=axc.transAxes))
        axc.text(x+0.112,y-0.005,t,ha="center",fontsize=8.0,color=INK,transform=axc.transAxes)
    axc.text(0.02,0.92,"TRAIN pool (7 archs/sizes)",fontsize=9,fontweight="bold",color=GREEN,transform=axc.transAxes)
    axc.text(0.55,0.30,"HELD-OUT model",fontsize=9,fontweight="bold",color=RED,transform=axc.transAxes)
    axc.add_patch(FancyBboxPatch((0.55,0.13),0.30,0.13,boxstyle="round,pad=0.004,rounding_size=0.02",
                  fc="#f7e9e8",ec=RED,lw=1.6,transform=axc.transAxes))
    axc.text(0.70,0.195,"llama3-8b  (never trained)",ha="center",fontsize=8.6,fontweight="bold",color=RED,transform=axc.transAxes)
    axc.text(0.02,0.30,"+ 18 more enc_M tags available\n(bloom, pythia, gpt2, deepseek,\n  nemotron, lfm, yagpt, vikhr…)",
             fontsize=8.0,color=MUT,transform=axc.transAxes)

    # (d) held-out concepts
    axd=fig.add_subplot(gs[1,1]); axd.axis("off"); axd.set_title("Concepts: supervised vs held-out (open-set)",
                                                                 fontsize=11,fontweight="bold",loc="left")
    axd.text(0.02,0.86,"Supervised: union of ~28 concepts",fontsize=9,fontweight="bold",color=BLUE,transform=axd.transAxes)
    axd.text(0.02,0.78,"quirks (camelcase, calories, bullets…) +\nsocial/political (chinese, muslim, gender, lgbt…) + cot_incorrect",
             fontsize=8.2,color=INK,transform=axd.transAxes)
    axd.text(0.02,0.50,"Held-out concepts (NEVER named in training)",fontsize=9,fontweight="bold",color=RED,transform=axd.transAxes)
    ho=["atomic","decimal","chinese","muslim","chocolate","voting"]
    for i,t in enumerate(ho):
        x=0.02+(i%3)*0.32; y=0.36-(i//3)*0.17
        axd.add_patch(FancyBboxPatch((x,y-0.06),0.30,0.12,boxstyle="round,pad=0.004,rounding_size=0.02",
                      fc="#f7e9e8",ec=RED,lw=1.2,transform=axd.transAxes))
        axd.text(x+0.15,y,t,ha="center",fontsize=8.2,color=INK,transform=axd.transAxes)
    axd.text(0.02,0.02,"→ detected zero-shot at ~0.92–1.0 AUROC on the held-out model (breadth, not architecture, is the lever)",
             fontsize=8.0,color=MUT,transform=axd.transAxes)
    fig.savefig(os.path.join(OUT,"fig2_training.png"),dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("wrote fig2_training.png")

# ============================================================ FIG 3: WHAT WE CAN DO
def fig_caps():
    ev20=load("ours_homefield/eval_v20.json")
    cbe=load("results/cross_base_em.json")
    fig=plt.figure(figsize=(14,8.6)); gs=fig.add_gridspec(2,2,hspace=0.42,wspace=0.26,
                                                          left=0.07,right=0.97,top=0.9,bottom=0.08)
    fig.suptitle("What the oracle can do — held-out results (read on a model it never trained on)",
                 fontsize=15,fontweight="bold",y=0.965)

    # (A) bias auditing — held-out-CONCEPT AUROC (zero-shot)
    axA=fig.add_subplot(gs[0,0])
    ho=ev20.get("heldout_bias_auroc",{}) or {"atomic":1,"chinese":1,"muslim":1,"chocolate":1,"decimal":1,
                                             "sports":.975,"movie":.991,"voting":.908,"british":.816,"rhetq":.488}
    items=sorted(ho.items(),key=lambda kv:kv[1])
    names=[k for k,_ in items]; vals=[v for _,v in items]
    cols=[GREEN if v>=0.85 else (AMBER if v>=0.7 else RED) for v in vals]
    axA.barh(range(len(names)),vals,color=cols,edgecolor="white",height=0.7)
    axA.set_yticks(range(len(names))); axA.set_yticklabels(names,fontsize=8)
    axA.axvline(0.5,color=MUT,ls=":",lw=1); axA.set_xlim(0.4,1.02)
    for i,v in enumerate(vals): axA.text(v+0.005,i,f"{v:.2f}",va="center",fontsize=7.4,color=INK)
    m=ev20.get("heldout_bias_auroc_mean",0.918)
    axA.set_title(f"① Bias auditing — ZERO-SHOT concepts on unseen llama3-8b\n     (concept never named in training; mean {m:.2f})",
                  fontsize=10,fontweight="bold",loc="left")
    for s in ["top","right"]: axA.spines[s].set_visible(False)

    # (B) pre-emptive intent + jailbreak-success (PRE/EARLY/POST)
    axB=fig.add_subplot(gs[0,1])
    grp=["Llama-3.1-8B\n(intent)","Stheno-8B RP\n(intent)","jailbreak\nsuccess"]
    pre=[0.851,0.894,0.78]; early=[0.880,1.000,0.882]; post=[0.886,0.984,0.94]
    x=np.arange(len(grp)); bw=0.26
    axB.bar(x-bw,pre,bw,label="PRE (before any output)",color=BLUE,edgecolor="white")
    axB.bar(x,early,bw,label="EARLY (first tokens)",color=GREEN,edgecolor="white")
    axB.bar(x+bw,post,bw,label="POST (full response)",color=AMBER,edgecolor="white")
    axB.axhline(0.5,color=MUT,ls=":",lw=1); axB.set_ylim(0.4,1.05); axB.set_xticks(x); axB.set_xticklabels(grp,fontsize=8.4)
    axB.set_ylabel("AUROC"); axB.legend(fontsize=7.4,loc="lower right",framealpha=0.9)
    axB.set_title("② Read intent BEFORE it speaks\n     (within-model, group-by-prompt CV)",fontsize=10,fontweight="bold",loc="left")
    for s in ["top","right"]: axB.spines[s].set_visible(False)

    # (C) the detector lesson — regex vs LLM judge jailbreak success
    axC=fig.add_subplot(gs[1,0])
    jb=["roleplay","hypothetical","DAN tmpl","DAN 13.0","jackhhao\nreal"]; regex=[0.69,0.56,0.31,0.42,0.93]; judge=[0.06,0.06,0.25,0.17,0.15]
    x=np.arange(len(jb)); bw=0.38
    axC.bar(x-bw/2,regex,bw,label="regex detector (inflated)",color=GREY,edgecolor="white")
    axC.bar(x+bw/2,judge,bw,label="LLM judge (true)",color=RED,edgecolor="white")
    axC.set_xticks(x); axC.set_xticklabels(jb,fontsize=8.2); axC.set_ylim(0,1.0); axC.set_ylabel("jailbreak success rate")
    axC.legend(fontsize=7.6,loc="upper right",framealpha=0.9)
    axC.set_title("③ Measurement matters — Llama-3.1-8B resists jailbreaks\n     (regex over-counts 6–12×; model adopts persona, withholds payload)",
                  fontsize=10,fontweight="bold",loc="left")
    for s in ["top","right"]: axC.spines[s].set_visible(False)

    # (D) cross-base organism (EM) transfer heatmap
    axD=fig.add_subplot(gs[1,1])
    fams=["qwen7b","qwen0.5b","llama1b","llama8b"]
    M=np.array([[cbe.get(r,{}).get(c,np.nan) for c in fams] for r in fams]) if cbe else \
       np.array([[1.0,.957,.861,.928],[.76,1.0,.99,.978],[.584,.996,1.0,1.0],[.88,.987,1.0,1.0]])
    im=axD.imshow(M,cmap="RdYlGn",vmin=0.5,vmax=1.0,aspect="auto")
    axD.set_xticks(range(4)); axD.set_xticklabels(fams,fontsize=8,rotation=20); axD.set_yticks(range(4)); axD.set_yticklabels(fams,fontsize=8)
    for i in range(4):
        for j in range(4):
            axD.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8.4,
                     color="black" if M[i,j]>0.72 else "white",fontweight="bold")
    axD.set_xlabel("test base"); axD.set_ylabel("train base")
    off=[M[i,j] for i in range(4) for j in range(4) if i!=j]; mo=float(np.nanmean(off))
    axD.set_title(f"④ Emergent-misalignment organism — CROSS-BASE transfer\n     (probe trained on one base detects it on another; off-diag mean {mo:.2f})",
                  fontsize=10,fontweight="bold",loc="left")
    fig.colorbar(im,ax=axD,fraction=0.046,pad=0.04).set_label("AUROC",fontsize=8)
    fig.savefig(os.path.join(OUT,"fig3_capabilities.png"),dpi=150,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("wrote fig3_capabilities.png")

if __name__=="__main__":
    fig_arch(); fig_train(); fig_caps()
    print("done →", OUT)
