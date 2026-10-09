import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
base = "results/phase3/xai/mode_ablation/mode_ablation_{}_k3.json"
folds = [("fold0","Fold 0"),("fold1","Fold 1"),("fold2_w512","Fold 2"),("fold3","Fold 3"),("fold4","Fold 4")]
BORDER = {("fold2_w512",0)}  # cleared the floor by 0.0004; not counted (Update 15)
NEED, BORD, NOT = "#2f6db5", "#9fbde0", "#d9d9d9"
fig, ax = plt.subplots(figsize=(8.2, 3.6))
for i,(tag,name) in enumerate(folds):
    d = json.load(open(base.format(tag)))
    ms = sorted(d["modes"], key=lambda m: m["mode"])
    left = 0
    y = len(folds)-1-i
    for m in ms:
        n = m["n_edges"]; v = m["verdict"]
        if (tag,m["mode"]) in BORDER: c, h = BORD, "//"
        elif v["claimable"]: c, h = NEED, None
        else: c, h = NOT, None
        ax.barh(y, n, left=left, color=c, hatch=h, edgecolor="white", linewidth=1.5, height=0.62)
        txt = f"{n}"
        if v["claimable"] and (tag,m["mode"]) not in BORDER:
            txt += f"\n−{v['mode_drop']:.3f}"
        ax.text(left+n/2, y, txt, ha="center", va="center", fontsize=8.5,
                color="white" if c==NEED else "#333")
        left += n
    base_auc = ms[0]["verdict"]["baseline_auc"]
    ax.text(79, y, f"AUC {base_auc:.2f}", va="center", fontsize=8.5, color="#555")
ax.set_yticks(range(len(folds))); ax.set_yticklabels([n for _,n in folds][::-1])
ax.set_xlim(0, 88); ax.set_xticks([0,20,40,60,77])
ax.set_xlabel("Top 77 delayed-signal connections per network, sorted into 3 groups by curve shape")
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=NEED,label="Needed (beats both controls by > 0.0092)"),
                   Patch(facecolor=BORD,hatch="//",edgecolor="white",label="Borderline (by 0.0004), not counted"),
                   Patch(color=NOT,label="Not needed")],
          loc="upper center", bbox_to_anchor=(0.45,1.17), ncol=3, frameon=False, fontsize=8)
ax.set_title("Numbers: group size, and AUC drop when a needed group is deleted", fontsize=9, color="#555", loc="left", pad=26)
plt.tight_layout()
out = "results/figures/research_update_assets/ru3_group_deletion.png"
plt.savefig(out, dpi=200, bbox_inches="tight"); print(out)
