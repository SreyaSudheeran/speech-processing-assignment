"""Evaluate the best text-only checkpoint on the test split."""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from common.train_utils import *
from common.models import SpeechModel, TextModel, FusionModel

if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    ck0 = torch.load(os.path.join(args.results_dir, "text", "best.pt"), map_location="cpu", weights_only=False)
    args.text_encoder = ck0["text_info"]["encoder"]          # exactly what training used
    D = prepare(args, need_speech=False, need_text=True)
    model = TextModel(D["text_info"]); ck = load_ckpt(model, args, "text")
    evaluate_and_report(model, D, args, "text", n_params=ck["n_params"])
