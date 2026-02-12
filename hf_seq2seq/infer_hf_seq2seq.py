"""
Inferenca za model istreniran skriptom `hf_seq2seq/train_hf_seq2seq.py`.
Za sažimanje: --sentence je dijalog (ili drugi ulaz), izlaz je rezime.
Za prevod: --sentence je rečenica na izvornom jeziku, izlaz je prevod.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import os
import sys

# Dodaj root projekta na sys.path da bismo mogli da uvezemo `model` i train modul
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from tokenizers import Tokenizer

from model import build_transformer
from hf_seq2seq.train_hf_seq2seq import greedy_decode  # koristimo istu funkciju


def find_latest_checkpoint(weights_dir: Path) -> Optional[Path]:
    # Prvo probaj best.pt (najbolji po validacionom loss-u), pa poslednji epoch
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(weights_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Ne postoji run_config.json u run folderu. Pokreni prvo trening.")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Novi run koristi source_column / target_column; stari run može imati source_language / target_language
    input_label = cfg.get("source_column", cfg.get("source_language", "input"))
    output_label = cfg.get("target_column", cfg.get("target_language", "output"))
    context_size = int(args.max_length) if args.max_length is not None else int(cfg["context_size"])

    tokenizers_dir = run_dir / "tokenizers"
    weights_dir = run_dir / "weights"

    src_tok_path = tokenizers_dir / "tokenizer_src.json"
    tgt_tok_path = tokenizers_dir / "tokenizer_tgt.json"
    if not src_tok_path.exists() or not tgt_tok_path.exists():
        raise FileNotFoundError("Ne postoje tokenizer fajlovi u run folderu. Pokreni prvo trening.")

    src_tok = Tokenizer.from_file(str(src_tok_path))
    tgt_tok = Tokenizer.from_file(str(tgt_tok_path))

    ckpt = find_latest_checkpoint(weights_dir)
    if ckpt is None:
        raise FileNotFoundError("Ne postoji checkpoint u run folderu. Pokreni prvo trening.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(
        source_vocab_size=src_tok.get_vocab_size(),
        target_vocab_size=tgt_tok.get_vocab_size(),
        source_context_size=int(cfg["context_size"]),
        target_context_size=int(cfg["context_size"]),
        model_dimension=int(cfg["model_dimension"]),
        number_of_blocks=int(cfg["number_of_blocks"]),
        heads=int(cfg["heads"]),
        dropout=float(cfg["dropout"]),
        feed_forward_dimension=int(cfg["feed_forward_dimension"]),
    ).to(device)

    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # encoder_input = [SOS] src_ids [EOS] PAD...
    src_ids = src_tok.encode(args.sentence).ids
    sos_id = src_tok.token_to_id("[SOS]")
    eos_id = src_tok.token_to_id("[EOS]")
    pad_id = src_tok.token_to_id("[PAD]")

    enc_pad = context_size - len(src_ids) - 2
    if enc_pad < 0:
        raise ValueError(f"Rečenica je predugačka za context_size={context_size}. Probaj veći --max-length.")

    encoder_input = torch.tensor([sos_id] + src_ids + [eos_id] + [pad_id] * enc_pad, dtype=torch.int64).unsqueeze(0).to(device)
    pad_tensor = torch.tensor([pad_id], dtype=torch.int64).to(device)
    encoder_mask = (encoder_input != pad_tensor).unsqueeze(0).unsqueeze(0).int()

    out_ids = greedy_decode(model, encoder_input, encoder_mask, tgt_tok, context_size, device)
    output_text = tgt_tok.decode(out_ids.detach().cpu().numpy())

    print(f"Input ({input_label}): {args.sentence[:200]}{'...' if len(args.sentence) > 200 else ''}")
    print(f"Output ({output_label}): {output_text}")


if __name__ == "__main__":
    main()

