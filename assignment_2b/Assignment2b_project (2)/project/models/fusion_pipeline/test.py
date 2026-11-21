"""Evaluate fusion model + modality ablations (speech zeroed / text zeroed) on the test split."""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from common.train_utils import *
from common.models import SpeechModel, TextModel, FusionModel

import json
if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser()).parse_args()
    set_seed(args.seed)
    ck0 = torch.load(os.path.join(args.results_dir, "fusion", "best.pt"), map_location="cpu", weights_only=False)
    args.text_encoder = ck0["text_info"]["encoder"]
    D = prepare(args, need_speech=True, need_text=True)
    model = FusionModel(D["text_info"], in_dim=D["sp"].shape[-1]); ck = load_ckpt(model, args, "fusion")
    res = [evaluate_and_report(model, D, args, "fusion", n_params=ck["n_params"])]
    model.zero_text = True;  res.append(evaluate_and_report(model, D, args, "fusion", tag="_speech_only_input"))
    model.zero_text = False; model.zero_speech = True
    res.append(evaluate_and_report(model, D, args, "fusion", tag="_text_only_input"))
    with open(os.path.join(args.results_dir, "fusion", "ablation.json"), "w") as f:
        json.dump(res, f, indent=1)
