"""Builds the comparison tables + analysis plots from the three pipelines' test outputs.

Run AFTER all three test.py scripts:  python analyze.py
Writes to Results/comparison/:
  accuracy_table.csv/.md, per_emotion_f1.csv, fusion_vs_speech_delta.csv, modality_ablation.csv,
  tsne_temporal.png, tsne_contextual.png, tsne_fusion.png, tsne_all.png, separability.csv, failure_cases.md
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
ROOT = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--results_dir", default=os.path.join(ROOT, "Results"))
args = p.parse_args()
R = args.results_dir
OUT = os.path.join(R, "comparison"); os.makedirs(OUT, exist_ok=True)

names = {"speech": "Speech-only", "text": "Text-only", "fusion": "Multimodal (fusion)"}
metrics = {k: json.load(open(os.path.join(R, k, "metrics.json"))) for k in names}

# ---- 1. accuracy table -------------------------------------------------------------
tab = pd.DataFrame([{"Model": names[k], "Test accuracy": round(100 * m["test_accuracy"], 2),
                     "Macro-F1": round(100 * m["test_macro_f1"], 2), "Params": m.get("n_params"),
                     "Test samples": m["n_test"]} for k, m in metrics.items()])
tab.to_csv(os.path.join(OUT, "accuracy_table.csv"), index=False)
open(os.path.join(OUT, "accuracy_table.md"), "w").write(tab.to_markdown(index=False))
print(tab.to_string(index=False))
plt.figure(figsize=(5, 3.4))
plt.bar(tab.Model, tab["Test accuracy"], color=["#3b6ea8", "#c9873a", "#4c9a6a"])
plt.axhline(100 / 7, ls="--", c="gray", lw=1); plt.text(2.45, 100 / 7 + 1.5, "chance", ha="right", color="gray", fontsize=8)
plt.ylabel("test accuracy (%)"); plt.ylim(0, 105); plt.xticks(rotation=12)
for i, v in enumerate(tab["Test accuracy"]): plt.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "accuracy_bar.png"), dpi=140); plt.close()

# ---- 2. per-emotion F1 + fusion gain ------------------------------------------------
f1 = pd.DataFrame({names[k]: pd.read_csv(os.path.join(R, k, "per_class_report.csv"), index_col=0)["f1-score"].reindex(EMOTIONS)
                   for k in names}).round(4)
f1.to_csv(os.path.join(OUT, "per_emotion_f1.csv"))
delta = (f1["Multimodal (fusion)"] - f1["Speech-only"]).round(4).rename("fusion_minus_speech_F1")
delta.to_csv(os.path.join(OUT, "fusion_vs_speech_delta.csv"))
print("\nPer-emotion F1:\n", f1.to_string(), "\n\nFusion - speech:\n", delta.to_string())
fig, ax = plt.subplots(figsize=(7, 3.4)); f1.plot.bar(ax=ax, width=0.8); ax.set_ylabel("F1"); ax.set_title("Per-emotion F1 (test)")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(os.path.join(OUT, "per_emotion_f1.png"), dpi=140); plt.close()

# ---- 3. modality ablation of the fusion model --------------------------------------
ab_path = os.path.join(R, "fusion", "ablation.json")
if os.path.exists(ab_path):
    ab = pd.DataFrame(json.load(open(ab_path)))[["model", "test_accuracy", "test_macro_f1"]]
    ab["model"] = ab["model"].map({"fusion": "fusion (both inputs)", "fusion_speech_only_input": "fusion, text zeroed",
                                   "fusion_text_only_input": "fusion, speech zeroed"})
    ab.round(4).to_csv(os.path.join(OUT, "modality_ablation.csv"), index=False)
    print("\nAblation:\n", ab.to_string(index=False))

# ---- 4. representation separability (t-SNE + silhouette + kNN) ---------------------
blocks = [("speech", "temporal", "Temporal modelling (speech)"),
          ("text", "contextual", "Contextual modelling (text)"),
          ("fusion", "fusion", "Fusion block")]
fig_all, axes = plt.subplots(1, 3, figsize=(17, 5)); sep = []
cmap = plt.get_cmap("tab10")
for ax_all, (run, key, title) in zip(axes, blocks):
    z = np.load(os.path.join(R, run, "embeddings_test.npz")); X, y = z[key], z["y"]
    # t-SNE needs distinct points: tiny jitter guards against collapsed (identical) embeddings
    Xj = X + 1e-6 * np.random.RandomState(0).randn(*X.shape)
    E = TSNE(2, perplexity=min(30, len(X) // 4), init="pca", random_state=0).fit_transform(Xj)
    ztr = np.load(os.path.join(R, run, "embeddings_train.npz"))
    sil = float(silhouette_score(X, y))                                   # label-based, no classifier fitted
    knn = float(KNeighborsClassifier(5).fit(ztr[key], ztr["y"]).score(X, y))   # fit on TRAIN, score on TEST
    sep.append({"Representation": title, "Silhouette (emotion labels)": round(sil, 4), "kNN acc (fit train, eval test)": round(knn, 4)})
    for a in (ax_all, None):
        if a is None:
            fig, a = plt.subplots(figsize=(6, 5))
        for c, e in enumerate(EMOTIONS):
            m = y == c; a.scatter(E[m, 0], E[m, 1], s=14, color=cmap(c), label=e, alpha=0.85)
        a.set_title(f"{title}\nsilhouette={sil:.2f}, kNN={knn:.2f}"); a.set_xticks([]); a.set_yticks([])
        if a is not ax_all:
            a.legend(markerscale=1.5, fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"tsne_{key}.png"), dpi=150); plt.close(fig)
axes[0].legend(markerscale=1.5, fontsize=8)
fig_all.tight_layout(); fig_all.savefig(os.path.join(OUT, "tsne_all.png"), dpi=150); plt.close(fig_all)
sep = pd.DataFrame(sep); sep.to_csv(os.path.join(OUT, "separability.csv"), index=False); print("\n", sep.to_string(index=False))

# ---- 5. error analysis: most confident mistakes of speech & fusion models ----------
with open(os.path.join(OUT, "failure_cases.md"), "w") as f:
    for run in ("speech", "fusion", "text"):
        d = pd.read_csv(os.path.join(R, run, "test_predictions.csv"))
        bad = d[~d.correct].sort_values("confidence", ascending=False).head(5)
        f.write(f"### {names[run]}  ({(~d.correct).sum()} errors / {len(d)})\n\n")
        f.write(bad[["speaker", "word", "emotion", "pred", "confidence"]].rename(columns={"emotion": "true"}).round(3).to_markdown(index=False) + "\n\n")
        if run != "text":
            conf = d[~d.correct].groupby(["emotion", "pred"]).size().sort_values(ascending=False).head(5)
            f.write("Most frequent confusions (true -> predicted): " + "; ".join(f"{a}->{b} x{n}" for (a, b), n in conf.items()) + "\n\n")
        if run == "speech":
            f.write("Errors by speaker:\n\n" + d.groupby("speaker").correct.apply(lambda s: f"{(~s).sum()}/{len(s)} wrong").to_frame().to_markdown() + "\n\n")
print("\nAll comparison outputs written to", OUT)
