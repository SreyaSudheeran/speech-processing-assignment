"""Multimodal: speech temporal encoder + text contextual encoder -> gated fusion -> classifier (end-to-end)."""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from common.train_utils import *
from common.models import SpeechModel, TextModel, FusionModel

if __name__ == "__main__":
    p = train_args(argparse.ArgumentParser())
    p.add_argument("--modality_dropout", type=float, default=0.15)
    args = p.parse_args()
    set_seed(args.seed)
    D = prepare(args, need_speech=True, need_text=True)
    fit(FusionModel(D["text_info"], drop=args.dropout, modality_dropout=args.modality_dropout, in_dim=D["sp"].shape[-1]), D, args, "fusion")
