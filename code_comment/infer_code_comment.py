"""
Inferenca za model istreniran za Code → Docstring.
Ulaz: Python kod (npr. funkcija), izlaz: opis u prirodnom jeziku (docstring).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from tokenizers import Tokenizer

from model import build_transformer
from code_comment.train_code_comment import greedy_decode


def find_latest_checkpoint(weights_dir: Path) -> Optional[Path]:
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(weights_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Putanja do run foldera (npr. runs/code_comment)")
    parser.add_argument("--code", required=True, help="Python kod (funkcija) za koju želiš opis")
    parser.add_argument("--max-length", type=int, default=None, help="Maks dužina izlaza (default: iz config-a)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Nema run_config.json. Prvo pokreni trening.")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    context_size = int(args.max_length) if args.max_length is not None else int(cfg["context_size"])

    tokenizers_dir = run_dir / "tokenizers"
    weights_dir = run_dir / "weights"
    src_tok_path = tokenizers_dir / "tokenizer_src.json"
    tgt_tok_path = tokenizers_dir / "tokenizer_tgt.json"
    if not src_tok_path.exists() or not tgt_tok_path.exists():
        raise FileNotFoundError("Nema tokenizer fajlova. Prvo pokreni trening.")

    src_tok = Tokenizer.from_file(str(src_tok_path))
    tgt_tok = Tokenizer.from_file(str(tgt_tok_path))

    ckpt_path = find_latest_checkpoint(weights_dir)
    if ckpt_path is None:
        raise FileNotFoundError("Nema checkpoint-a. Prvo pokreni trening.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(
        source_vocab_size=src_tok.get_vocab_size(),
        target_vocab_size=tgt_tok.get_vocab_size(),
        source_context_size=context_size,
        target_context_size=context_size,
        model_dimension=int(cfg["model_dimension"]),
        number_of_blocks=int(cfg["number_of_blocks"]),
        heads=int(cfg["heads"]),
        dropout=float(cfg["dropout"]),
        feed_forward_dimension=int(cfg["feed_forward_dimension"]),
    ).to(device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    src_ids = src_tok.encode(args.code).ids
    sos_id = src_tok.token_to_id("[SOS]")
    eos_id = src_tok.token_to_id("[EOS]")
    pad_id = src_tok.token_to_id("[PAD]")
    enc_pad = context_size - len(src_ids) - 2
    if enc_pad < 0:
        raise ValueError(f"Kod je predugačak za context_size={context_size}. Povećaj --max-length.")

    encoder_input = (
        torch.tensor([sos_id] + src_ids + [eos_id] + [pad_id] * enc_pad, dtype=torch.int64)
        .unsqueeze(0)
        .to(device)
    )
    encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()

    out_ids = greedy_decode(model, encoder_input, encoder_mask, tgt_tok, context_size, device)
    docstring = tgt_tok.decode(out_ids.detach().cpu().numpy())

    print("Kod:")
    print(args.code[:500] + ("..." if len(args.code) > 500 else ""))
    print("\nOpis (docstring):")
    print(docstring)


if __name__ == "__main__":
    main()
