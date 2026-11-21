"""Evaluate the best speech-only checkpoint on the (word-disjoint) test split."""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from common.train_utils import *
from common.models import SpeechModel, TextModel, FusionModel

if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    D = prepare(args, need_speech=True, need_text=False)
    model = SpeechModel(in_dim=D["sp"].shape[-1]); ck = load_ckpt(model, args, "speech")
    evaluate_and_report(model, D, args, "speech", n_params=ck["n_params"])
