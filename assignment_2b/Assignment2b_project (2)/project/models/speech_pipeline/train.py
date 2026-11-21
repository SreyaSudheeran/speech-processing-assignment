"""Speech-only: preprocessing -> log-mel -> CNN+BiGRU temporal model -> classifier."""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from common.train_utils import *
from common.models import SpeechModel, TextModel, FusionModel

if __name__ == "__main__":
    args = train_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    D = prepare(args, need_speech=True, need_text=False)
    fit(SpeechModel(drop=args.dropout, in_dim=D["sp"].shape[-1]), D, args, "speech")
