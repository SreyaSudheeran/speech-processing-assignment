"""Data loading, preprocessing and feature extraction for TESS.

TESS file naming:  <SPEAKER>_<word>_<emotion>.wav   e.g. OAF_back_angry.wav, YAF_base_ps.wav
Every clip is the carrier phrase "Say the word <word>" spoken by one of two actresses
(OAF = older, YAF = younger) in 7 emotions. The Kaggle copy ships no transcript files,
so the transcript is rebuilt from the filename: "Say the word <word>".
"""
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]  # ps = pleasant surprise
EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)

# ---- speech preprocessing / feature constants ----
SR = 16000            # sampling rate (Hz)
MAX_SEC = 3.0         # clips are cropped / zero-padded to this length after silence trimming
TRIM_DB = 30          # silence trimming threshold
N_FFT, WIN, HOP, N_MELS = 512, 400, 160, 64   # 25 ms window, 10 ms hop, 64 mel bands
MAX_FRAMES = int(MAX_SEC * SR / HOP) + 1      # 301 frames

_PREFIX = {"ang": "angry", "dis": "disgust", "fea": "fear", "hap": "happy",
           "neu": "neutral", "ple": "ps", "sur": "ps", "ps": "ps", "sad": "sad"}


def _norm_emotion(token, folder):
    for t in (token.lower(), folder.lower().split("_")[-1], folder.lower()):
        for k, v in _PREFIX.items():
            if t.startswith(k):
                return v
    return None


# --------------------------------------------------------------------------------------
# Metadata + split
# --------------------------------------------------------------------------------------
def find_data_dir(data_dir=None):
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    for cand in ["/kaggle/input", "./data", "../data"]:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("Could not find TESS. Pass --data_dir pointing at the folder with the .wav files.")


def build_metadata(data_dir=None, seed=42, val_frac=0.15, test_frac=0.15, split="word", test_speaker="YAF"):
    """Scan for TESS wavs -> DataFrame[path, speaker, word, emotion, label, text, split].

    split="word"   : WORD-DISJOINT 70/15/15 (both speakers in every split)
    split="speaker": SPEAKER-INDEPENDENT: test = every clip of `test_speaker`; train/val come from the other
                     speaker (val = 15% of its words, word-disjoint from train).
    The default word split is WORD-DISJOINT (all 14 clips of a word go to the same split): the model is
    tested on words it never saw, so it cannot memorise lexical items. Each split stays
    perfectly class balanced because every word occurs in all 7 emotions.
    """
    root = find_data_dir(data_dir)
    paths = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
                   + glob.glob(os.path.join(root, "**", "*.WAV"), recursive=True))
    rows, seen = [], set()
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split("_")
        if len(parts) < 3 or parts[0].upper() not in ("OAF", "YAF"):
            continue
        emo = _norm_emotion(parts[-1], os.path.basename(os.path.dirname(p)))
        if emo is None:
            continue
        spk, word = parts[0].upper(), "_".join(parts[1:-1]).lower()
        key = (spk, word, emo)
        if key in seen:                      # Kaggle copy contains a nested duplicate folder
            continue
        seen.add(key)
        rows.append(dict(path=p, speaker=spk, word=word, emotion=emo, label=EMO2ID[emo],
                         text=f"Say the word {word}"))
    if not rows:
        raise RuntimeError(f"No TESS wav files found under {root}")
    df = pd.DataFrame(rows).sort_values(["word", "speaker", "emotion"]).reset_index(drop=True)

    words = sorted(df.word.unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(words)
    n_test, n_val = int(round(len(words) * test_frac)), int(round(len(words) * val_frac))
    if split == "speaker":
        val_w = set(words[:n_val])
        df["split"] = np.where(df.speaker == test_speaker.upper(), "test",
                               np.where(df.word.isin(val_w), "val", "train"))
    else:
        test_w, val_w = set(words[:n_test]), set(words[n_test:n_test + n_val])
        df["split"] = df.word.map(lambda w: "test" if w in test_w else ("val" if w in val_w else "train"))
    return df


# --------------------------------------------------------------------------------------
# Speech: preprocessing + log-mel features
# --------------------------------------------------------------------------------------
def load_wave(path):
    """Resample to 16 kHz mono, trim leading/trailing silence, peak-normalise, crop to MAX_SEC."""
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)
    if len(y) < 400:
        y = np.pad(y, (0, 400 - len(y)))
    y = 0.95 * y / (np.max(np.abs(y)) + 1e-8)
    return y[: int(MAX_SEC * SR)].astype(np.float32)


def log_mel(y):
    """(time_steps x 64) log-mel spectrogram in dB."""
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                                       n_mels=N_MELS, fmin=20, fmax=SR // 2, power=2.0)
    return librosa.power_to_db(m, ref=1.0, top_db=80.0).T.astype(np.float32)


def _one(path):
    return log_mel(load_wave(path))


def _hash(df):
    return hashlib.md5("|".join(df.path.map(os.path.basename)).encode()).hexdigest()[:10]


WAVLM_NAME = "microsoft/wavlm-base-plus"
WAVLM_LAYERS = (4, 9)       # average hidden states of layers 4..8 (mid layers carry most paralinguistic/emotion info)
WAVLM_FRAMES = int(MAX_SEC * 50) - 1     # 20 ms frames -> 149 for 3 s


def _wavlm_features(df, device):
    """Frozen WavLM-base+ -> (N, WAVLM_FRAMES, 768): mean of mid layers, one utterance at a time (no padding artefacts)."""
    import torch
    from transformers import AutoModel
    net = AutoModel.from_pretrained(WAVLM_NAME).to(device).eval()
    feats = []
    with torch.no_grad():
        for k, p in enumerate(df.path):
            y = load_wave(p)
            y = (y - y.mean()) / (y.std() + 1e-7)                       # WavLM expects zero-mean/unit-var input
            hs = net(torch.from_numpy(y)[None].to(device), output_hidden_states=True).hidden_states
            h = torch.stack(hs[WAVLM_LAYERS[0]:WAVLM_LAYERS[1]]).mean(0)[0].cpu().numpy()
            feats.append(h)
            if k % 500 == 0:
                print(f"[data] wavlm {k}/{len(df)}")
    return feats


def get_speech_features(df, cache_dir="cache", feat="logmel", spk_norm=False, device="cpu"):
    """Returns X (N, T, D) standardised (padding = 0), lens (N,).
    feat='logmel': 64-band log-mel (D=64, 10 ms hop)   feat='wavlm': frozen WavLM-base+ (D=768, 20 ms hop)
    Standardisation: with TRAIN-set statistics by default; with spk_norm=True per SPEAKER (each speaker's own
    mean/std over all of that speaker's clips, labels never used) -> removes speaker-specific pitch/timbre offsets.
    Note spk_norm uses the unlabeled test speaker's audio statistics (transductive), report it as such."""
    from joblib import Parallel, delayed
    os.makedirs(cache_dir, exist_ok=True)
    T = MAX_FRAMES if feat == "logmel" else WAVLM_FRAMES
    f = os.path.join(cache_dir, f"speech_{feat}_{_hash(df)}.npz")
    if os.path.exists(f):
        z = np.load(f); raw, lens = z["raw"].astype(np.float32), z["lens"]
    else:
        print(f"[data] extracting {feat} features for {len(df)} clips ...")
        if feat == "logmel":
            feats = Parallel(n_jobs=-1)(delayed(_one)(p) for p in df.path)
        else:
            feats = _wavlm_features(df, device)
        D = feats[0].shape[1]
        lens = np.array([min(len(x), T) for x in feats], dtype=np.int64)
        raw = np.zeros((len(df), T, D), dtype=np.float32)
        for i, x in enumerate(feats):
            raw[i, : lens[i]] = x[: lens[i]]
        np.savez(f, raw=raw.astype(np.float16) if feat == "wavlm" else raw, lens=lens)
    mask = np.arange(T)[None, :] < lens[:, None]
    X = np.zeros_like(raw)
    if spk_norm:
        for spk in df.speaker.unique():
            m = (df.speaker == spk).values
            v = raw[m][mask[m]]
            X[m] = (raw[m] - v.mean(0)) / (v.std(0) + 1e-5)
    else:
        tr = (df.split == "train").values
        v = raw[tr][mask[tr]]
        X = (raw - v.mean(0)) / (v.std(0) + 1e-5)
    X[~mask] = 0.0
    return X.astype(np.float32), lens


# --------------------------------------------------------------------------------------
# Text: cleaning, tokenisation, features
# --------------------------------------------------------------------------------------
def clean_text(t):
    t = re.sub(r"[^a-z' ]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def get_text_features(df, encoder="distilbert", max_len=12, device="cpu", allow_fallback=False):
    """Returns (X, mask, info).
    - distilbert: X float (N, L, 768) frozen contextual token embeddings  (tokens x features)
    - scratch   : X long  (N, L) word ids; the model learns its own embedding table
    mask is bool (N, L) marking real tokens. Raises if DistilBERT can't be loaded, unless allow_fallback
    (then it uses 'scratch'); the encoder actually used is returned in info['encoder']."""
    texts = [clean_text(t) for t in df.text]
    if encoder == "distilbert":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            net = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()
            feats, masks = [], []
            with torch.no_grad():
                for i in range(0, len(texts), 128):
                    b = tok(texts[i:i + 128], padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
                    feats.append(net(**b).last_hidden_state.cpu().numpy())
                    masks.append(b["attention_mask"].bool().cpu().numpy())
            return np.concatenate(feats).astype(np.float32), np.concatenate(masks), {"encoder": "distilbert", "dim": 768}
        except Exception as e:  # noqa
            if not allow_fallback:
                raise RuntimeError(f"DistilBERT could not be loaded ({type(e).__name__}: {e}). Turn Internet ON in Kaggle, "
                                   "or rerun with --text_encoder scratch (or --allow_fallback) to use learned embeddings.") from e
            print(f"[data] DistilBERT unavailable ({type(e).__name__}); falling back to scratch embeddings")
    # scratch: word-level vocab from TRAIN texts only (unseen test words -> <unk>)
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in [clean_text(t) for t in df.text[df.split == "train"]]:
        for w in t.split():
            vocab.setdefault(w, len(vocab))
    L = 6
    ids = np.zeros((len(texts), L), dtype=np.int64)
    for i, t in enumerate(texts):
        for j, w in enumerate(t.split()[:L]):
            ids[i, j] = vocab.get(w, 1)
    return ids, ids != 0, {"encoder": "scratch", "vocab_size": len(vocab)}
