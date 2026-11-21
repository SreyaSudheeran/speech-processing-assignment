"""Model blocks: speech temporal encoder, text contextual encoder, gated fusion, classifier."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data import N_MELS, NUM_CLASSES


class AttnPool(nn.Module):
    """Masked attention pooling over time/tokens."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x, mask):
        a = self.w(x).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(a, dim=1)
        return (a.unsqueeze(-1) * x).sum(1)


def _bigru(rnn, x, lens):
    T = x.size(1)
    p = pack_padded_sequence(x, lens.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
    out, _ = rnn(p)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
    return out


class SpeechEncoder(nn.Module):
    """Temporal modelling: 2x Conv1d (local spectro-temporal patterns, 4x downsample) -> BiGRU -> attention pooling."""
    def __init__(self, in_dim=N_MELS, ch=128, hid=128, emb=256, drop=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU(), nn.MaxPool1d(2),
            nn.Dropout(drop))
        self.rnn = nn.GRU(ch, hid, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * hid)
        self.proj = nn.Sequential(nn.Linear(2 * hid, emb), nn.GELU(), nn.Dropout(drop))
        self.emb_dim = emb

    def forward(self, x, lens):                      # x: (B, T, F)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)   # (B, T/4, ch)
        T = h.size(1)
        l4 = ((lens + 3) // 4).clamp(1, T)
        mask = torch.arange(T, device=x.device)[None, :] < l4[:, None]
        h = _bigru(self.rnn, h, l4)
        return self.proj(self.pool(h, mask))


class TextEncoder(nn.Module):
    """Contextual modelling over token features: projection -> BiGRU -> attention pooling.
    Accepts float (B, L, D) frozen DistilBERT features, or long (B, L) ids (learned embedding)."""
    def __init__(self, in_dim=768, vocab_size=None, hid=128, emb=256, drop=0.3):
        super().__init__()
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
            in_dim = 128
        else:
            self.embed = None
        self.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(in_dim, 256), nn.GELU())
        self.rnn = nn.GRU(256, hid, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * hid)
        self.proj = nn.Sequential(nn.Linear(2 * hid, emb), nn.GELU(), nn.Dropout(drop))
        self.emb_dim = emb

    def forward(self, x, mask):
        if self.embed is not None:
            x = self.embed(x)
        h = self.inp(x)
        h = _bigru(self.rnn, h, mask.sum(1))
        return self.proj(self.pool(h, mask))


class GatedFusion(nn.Module):
    """A learned sigmoid gate decides, per dimension, how much to trust speech vs text;
    the gated mix is concatenated with both raw embeddings and projected to the fused representation."""
    def __init__(self, emb=256, out=256, drop=0.3):
        super().__init__()
        self.gate = nn.Linear(2 * emb, emb)
        self.mix = nn.Sequential(nn.Linear(3 * emb, out), nn.GELU(), nn.Dropout(drop))
        self.emb_dim = out

    def forward(self, s, t):
        g = torch.sigmoid(self.gate(torch.cat([s, t], -1)))
        return self.mix(torch.cat([g * s + (1 - g) * t, s, t], -1))


class Classifier(nn.Module):
    def __init__(self, emb=256, hid=128, n_cls=NUM_CLASSES, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, n_cls))

    def forward(self, z):
        return self.net(z)


# ---- full pipelines. forward(batch) -> (logits, {block_name: representation}) ----
class SpeechModel(nn.Module):
    def __init__(self, emb=256, drop=0.3, in_dim=N_MELS):
        super().__init__()
        self.speech = SpeechEncoder(in_dim=in_dim, emb=emb, drop=drop)
        self.clf = Classifier(emb, drop=drop)

    def forward(self, b):
        z = self.speech(b["sp"], b["sp_len"])
        return self.clf(z), {"temporal": z}


class TextModel(nn.Module):
    def __init__(self, text_info, emb=256, drop=0.3):
        super().__init__()
        self.text = TextEncoder(in_dim=text_info.get("dim", 768), vocab_size=text_info.get("vocab_size"),
                                emb=emb, drop=drop)
        self.clf = Classifier(emb, drop=drop)

    def forward(self, b):
        z = self.text(b["tx"], b["tx_mask"])
        return self.clf(z), {"contextual": z}


class FusionModel(nn.Module):
    def __init__(self, text_info, emb=256, drop=0.3, modality_dropout=0.15, in_dim=N_MELS):
        super().__init__()
        self.speech = SpeechEncoder(in_dim=in_dim, emb=emb, drop=drop)
        self.text = TextEncoder(in_dim=text_info.get("dim", 768), vocab_size=text_info.get("vocab_size"),
                                emb=emb, drop=drop)
        self.fusion = GatedFusion(emb, emb, drop)
        self.clf = Classifier(emb, drop=drop)
        self.p_md = modality_dropout
        self.zero_speech = False   # test-time ablation switches
        self.zero_text = False

    def forward(self, b):
        s = self.speech(b["sp"], b["sp_len"])
        t = self.text(b["tx"], b["tx_mask"])
        if self.training and self.p_md > 0:      # modality dropout: stops the net ignoring one stream
            r = torch.rand(s.size(0), device=s.device)
            s = s * (~(r < self.p_md / 2)).float().unsqueeze(1)
            t = t * (~((r >= self.p_md / 2) & (r < self.p_md))).float().unsqueeze(1)
        if self.zero_speech:
            s = torch.zeros_like(s)
        if self.zero_text:
            t = torch.zeros_like(t)
        z = self.fusion(s, t)
        return self.clf(z), {"temporal": s, "contextual": t, "fusion": z}
