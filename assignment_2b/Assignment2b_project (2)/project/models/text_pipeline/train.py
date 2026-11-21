"""Text-only: clean/tokenise -> DistilBERT token features -> BiGRU contextual model -> classifier."""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from common.train_utils import *
from common.models import SpeechModel, TextModel, FusionModel

if __name__ == "__main__":
    args = train_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    D = prepare(args, need_speech=False, need_text=True)
    fit(TextModel(D["text_info"], drop=args.dropout), D, args, "text")
