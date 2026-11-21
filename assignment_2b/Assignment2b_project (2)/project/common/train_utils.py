"""Datasets, training loop, evaluation, reporting (shared by all three pipelines)."""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from .data import (EMOTIONS, build_metadata, get_speech_features, get_text_features)


def add_common_args(p):
    p.add_argument("--data_dir", default=None, help="folder containing TESS .wav files (default: auto-detect under /kaggle/input)")
    p.add_argument("--results_dir", default=os.path.join(ROOT, "Results"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "cache"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--text_encoder", default="distilbert", choices=["distilbert", "scratch"])
    p.add_argument("--speech_feat", default="logmel", choices=["logmel", "wavlm"], help="wavlm = frozen WavLM-base+ (needs internet)")
    p.add_argument("--spk_norm", action="store_true", help="per-speaker feature standardisation (uses unlabeled speaker statistics)")
    p.add_argument("--allow_fallback", action="store_true", help="use scratch text embeddings if DistilBERT can't load (default: error)")
    p.add_argument("--split", default="word", choices=["word", "speaker"], help="word: unseen words; speaker: unseen speaker")
    p.add_argument("--test_speaker", default="YAF", choices=["OAF", "YAF"], help="held-out speaker when --split speaker")
    return p


def train_args(p):
    add_common_args(p)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    return p


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------------------
class TessDataset(Dataset):
    def __init__(self, idx, y, sp=None, sp_len=None, tx=None, tx_mask=None, augment=False):
        self.idx, self.y, self.sp, self.sp_len, self.tx, self.tx_mask, self.augment = idx, y, sp, sp_len, tx, tx_mask, augment

    def __len__(self):
        return len(self.idx)

    def _specaug(self, x, n):
        x = x.copy()
        x += np.random.randn(*x.shape).astype(np.float32) * 0.05                    # feature noise
        for _ in range(2):                                                          # time masks
            w = np.random.randint(0, 16)
            if n > w + 1:
                s = np.random.randint(0, n - w); x[s:s + w] = 0
        for _ in range(2):                                                          # freq masks
            w = np.random.randint(0, 8); s = np.random.randint(0, x.shape[1] - w); x[:n, s:s + w] = 0
        x[n:] = 0
        return x

    def __getitem__(self, i):
        j = self.idx[i]
        out = {"y": torch.tensor(self.y[j], dtype=torch.long), "i": torch.tensor(j)}
        if self.sp is not None:
            x, n = self.sp[j], int(self.sp_len[j])
            out["sp"] = torch.from_numpy(self._specaug(x, n) if self.augment else x)
            out["sp_len"] = torch.tensor(n)
        if self.tx is not None:
            out["tx"] = torch.from_numpy(self.tx[j])
            out["tx_mask"] = torch.from_numpy(self.tx_mask[j])
        return out


def prepare(args, need_speech, need_text):
    """Builds metadata/splits + features. Returns dict with df, features, loaders-factory."""
    os.makedirs(args.results_dir, exist_ok=True)
    df = build_metadata(args.data_dir, seed=args.seed, split=args.split, test_speaker=args.test_speaker)
    df.drop(columns=["path"]).to_csv(os.path.join(args.results_dir, "metadata_splits.csv"), index=False)
    print(df.groupby("split").size().to_dict(), "| speakers:", df.speaker.unique().tolist())
    D = {"df": df, "y": df.label.values, "sp": None, "sp_len": None, "tx": None, "tx_mask": None, "text_info": None}
    if need_speech:
        D["sp"], D["sp_len"] = get_speech_features(df, args.cache_dir, args.speech_feat, args.spk_norm, str(device()))
    if need_text:
        D["tx"], D["tx_mask"], D["text_info"] = get_text_features(df, args.text_encoder, device=str(device()), allow_fallback=args.allow_fallback)
    return D


def loader(D, split, batch_size, train=False):
    idx = np.where(D["df"].split.values == split)[0]
    ds = TessDataset(idx, D["y"], D["sp"], D["sp_len"], D["tx"], D["tx_mask"], augment=train and D["sp"] is not None)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=False, num_workers=0)


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


# ---------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    ys, probs, idxs, embs = [], [], [], {}
    for b in dl:
        b = to_dev(b, dev)
        logits, e = model(b)
        probs.append(torch.softmax(logits, -1).cpu().numpy()); ys.append(b["y"].cpu().numpy()); idxs.append(b["i"].cpu().numpy())
        for k, v in e.items():
            embs.setdefault(k, []).append(v.cpu().numpy())
    probs = np.concatenate(probs)
    return np.concatenate(ys), probs, np.concatenate(idxs), {k: np.concatenate(v) for k, v in embs.items()}


def fit(model, D, args, name):
    dev = device(); model.to(dev)
    out_dir = os.path.join(args.results_dir, name); os.makedirs(out_dir, exist_ok=True)
    tr, va = loader(D, "train", args.batch_size, True), loader(D, "val", args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] params: {n_par:,} | device: {dev}")
    best, bad, hist = (-1, 1e9), 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0, tl, correct, n = time.time(), 0.0, 0, 0
        for b in tr:
            b = to_dev(b, dev)
            logits, _ = model(b)
            loss = crit(logits, b["y"])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item() * len(b["y"]); correct += (logits.argmax(-1) == b["y"]).sum().item(); n += len(b["y"])
        yv, pv, _, _ = predict(model, va, dev)
        vloss = float(-np.log(pv[np.arange(len(yv)), yv] + 1e-9).mean()); vacc = float((pv.argmax(1) == yv).mean())
        hist.append(dict(epoch=ep, train_loss=tl / n, train_acc=correct / n, val_loss=vloss, val_acc=vacc))
        print(f"[{name}] ep {ep:02d} train {tl/n:.3f}/{correct/n:.3f} | val {vloss:.3f}/{vacc:.3f} | {time.time()-t0:.1f}s")
        if (vacc, -vloss) > (best[0], -best[1]):
            best, bad = (vacc, vloss), 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "text_info": D["text_info"], "val_acc": vacc,
                        "n_params": n_par}, os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{name}] early stop at epoch {ep}"); break
    h = pd.DataFrame(hist); h.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    _plot_history(h, out_dir, name)
    print(f"[{name}] best val acc {best[0]:.4f}")


def _plot_history(h, out_dir, name):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val"); ax[0].set_title(f"{name}: loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.train_acc, label="train"); ax[1].plot(h.epoch, h.val_acc, label="val"); ax[1].set_title(f"{name}: accuracy"); ax[1].legend()
    for a in ax: a.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130); plt.close()


# ---------------------------------------------------------------------------------------
def load_ckpt(model, args, name):
    ck = torch.load(os.path.join(args.results_dir, name, "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.to(device()).eval()
    return ck


def evaluate_and_report(model, D, args, name, tag="", n_params=None):
    """Runs the test split, writes metrics / confusion matrix / per-class report / embeddings / failure cases."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    out_dir = os.path.join(args.results_dir, name); os.makedirs(out_dir, exist_ok=True)
    y, probs, idx, embs = predict(model, loader(D, "test", args.batch_size), device())
    pred = probs.argmax(1)
    m = {"model": name + tag, "test_accuracy": float(accuracy_score(y, pred)),
         "test_macro_f1": float(f1_score(y, pred, average="macro")), "n_test": int(len(y)),
         "speech_feat": args.speech_feat, "spk_norm": args.spk_norm, "split": args.split + (f"(test={args.test_speaker})" if args.split == "speaker" else ""),
         "text_encoder": (D["text_info"] or {}).get("encoder")}
    print(json.dumps(m, indent=1))
    if tag == "":
        rep = pd.DataFrame(classification_report(y, pred, target_names=EMOTIONS, output_dict=True, zero_division=0)).T
        rep.to_csv(os.path.join(out_dir, "per_class_report.csv"))
        cm = confusion_matrix(y, pred, labels=range(len(EMOTIONS)))
        plt.figure(figsize=(5.5, 4.6))
        sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS, cbar=False)
        plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"{name}: test confusion (counts; colour = row-normalised)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=140); plt.close()
        np.savez(os.path.join(out_dir, "embeddings_test.npz"), y=y, idx=idx, **embs)
        ytr, _, itr, etr = predict(model, loader(D, "train", args.batch_size), device())   # no augmentation
        np.savez(os.path.join(out_dir, "embeddings_train.npz"), y=ytr, idx=itr, **etr)
        d = D["df"].iloc[idx].reset_index(drop=True)
        d["pred"] = [EMOTIONS[i] for i in pred]; d["confidence"] = probs.max(1); d["correct"] = pred == y
        d.drop(columns=["path"]).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
        d[~d.correct].sort_values("confidence", ascending=False).drop(columns=["path", "correct"]).head(20) \
            .to_csv(os.path.join(out_dir, "failure_cases.csv"), index=False)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({**m, "n_params": n_params}, f, indent=1)
    return m
