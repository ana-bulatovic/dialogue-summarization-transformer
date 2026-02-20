"""
Code → Docstring (opis koda). Podržani dataseti: CodeSearchNet Python, CM/codexglue_code2text_python.

Ulaz: Python kod (funkcija), izlaz: kratak opis u prirodnom jeziku (docstring).
Koristi isti encoder-decoder Transformer kao hf_seq2seq (model.py, dataset.causal_mask).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import causal_mask
from model import build_transformer


# CodeSearchNet Python: kolone func_code_string, func_documentation_string; split "valid"
DATASET_NAME = "code-search-net/code_search_net"
SOURCE_COLUMN = "func_code_string"
TARGET_COLUMN = "func_documentation_string"


def get_row_value(row: Dict[str, Any], key: str) -> str:
    """Vadi string iz reda. Podržava ugnježdene ključeve (npr. translation.en)."""
    if not key or "." not in key:
        val = row.get(key)
        return val if isinstance(val, str) else (str(val) if val is not None else "")
    part, sub = key.split(".", 1)
    val = row.get(part)
    if isinstance(val, dict):
        val = val.get(sub)
    return val if isinstance(val, str) else (str(val) if val is not None else "")


@dataclass(frozen=True)
class RunConfig:
    dataset_name: str
    dataset_config: Optional[str]
    source_column: str
    target_column: str
    context_size: int
    model_dimension: int
    number_of_blocks: int
    heads: int
    dropout: float
    feed_forward_dimension: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    seed: int
    max_train_samples: int
    max_val_samples: int
    max_test_samples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_hf_splits(
    dataset_name: str,
    dataset_config: Optional[str],
    seed: int,
    max_train: int,
    max_val: int,
    max_test: int,
    language_filter: Optional[str] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Učitava HF dataset i vraća (train, validation, test). CodeSearchNet koristi split 'valid'."""
    ds = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)

    if "train" in ds:
        train = ds["train"]
    else:
        train = ds[list(ds.keys())[0]]
    val = ds.get("validation") or ds.get("valid")
    test = ds.get("test")

    if language_filter and "language" in train.column_names:
        train = train.filter(lambda x: x.get("language") == language_filter)
        if val is not None:
            val = val.filter(lambda x: x.get("language") == language_filter)
        if test is not None:
            test = test.filter(lambda x: x.get("language") == language_filter)

    if val is None or test is None:
        tmp_size = min(max_val + max_test, max(1, int(0.2 * len(train))))
        split = train.train_test_split(test_size=tmp_size, seed=seed, shuffle=True)
        train = split["train"]
        tmp = split["test"]
        if len(tmp) <= 1:
            val = tmp
            test = tmp
        else:
            test_size = min(max_test, max(1, len(tmp) // 2))
            split2 = tmp.train_test_split(test_size=test_size, seed=seed, shuffle=True)
            val = split2["train"]
            test = split2["test"]

    train = train.select(range(min(max_train, len(train))))
    val = val.select(range(min(max_val, len(val))))
    test = test.select(range(min(max_test, len(test))))
    return train, val, test


def iter_sentences(dataset: Dataset, column_key: str) -> Iterable[str]:
    for x in dataset:
        yield get_row_value(x, column_key)


def build_or_load_tokenizer(
    path: Path,
    dataset: Dataset,
    column_key: str,
    force_rebuild: bool,
    min_frequency: int = 2,
    vocab_size: int = 50_000,
) -> Tokenizer:
    if path.exists() and not force_rebuild:
        return Tokenizer.from_file(str(path))
    ensure_dir(path.parent)
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=min_frequency,
        vocab_size=vocab_size,
    )
    tok.train_from_iterator(iter_sentences(dataset, column_key), trainer=trainer)
    tok.save(str(path))
    return tok


def filter_too_long(
    dataset: Dataset,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    source_column: str,
    target_column: str,
    context_size: int,
) -> Dataset:
    def ok(ex):
        s = source_tokenizer.encode(get_row_value(ex, source_column)).ids
        t = target_tokenizer.encode(get_row_value(ex, target_column)).ids
        return (len(s) <= context_size - 2) and (len(t) <= context_size - 1)

    return dataset.filter(ok)


class Seq2SeqDataset(TorchDataset):
    """Dataset parova (izvor, cilj) za encoder-decoder — kod → docstring."""

    def __init__(
        self,
        dataset: Dataset,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        source_column: str,
        target_column: str,
        context_size: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_col = source_column
        self.tgt_col = target_column
        self.context_size = context_size
        self.src_sos = torch.tensor([source_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.src_eos = torch.tensor([source_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.src_pad = torch.tensor([source_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.tgt_sos = torch.tensor([target_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.tgt_eos = torch.tensor([target_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.tgt_pad = torch.tensor([target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset[int(idx)]
        src_text = get_row_value(row, self.src_col)
        tgt_text = get_row_value(row, self.tgt_col)
        src_ids = self.source_tokenizer.encode(src_text).ids
        tgt_ids = self.target_tokenizer.encode(tgt_text).ids
        enc_pad = self.context_size - len(src_ids) - 2
        dec_pad = self.context_size - len(tgt_ids) - 1
        if enc_pad < 0 or dec_pad < 0:
            raise ValueError("Sekvenca predugačka za context_size.")
        encoder_input = torch.cat([
            self.src_sos, torch.tensor(src_ids, dtype=torch.int64), self.src_eos,
            torch.full((enc_pad,), self.src_pad.item(), dtype=torch.int64),
        ], dim=0)
        decoder_input = torch.cat([
            self.tgt_sos, torch.tensor(tgt_ids, dtype=torch.int64),
            torch.full((dec_pad,), self.tgt_pad.item(), dtype=torch.int64),
        ], dim=0)
        label = torch.cat([
            torch.tensor(tgt_ids, dtype=torch.int64), self.tgt_eos,
            torch.full((dec_pad,), self.tgt_pad.item(), dtype=torch.int64),
        ], dim=0)
        encoder_mask = (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.tgt_pad).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "source_text": src_text,
            "target_text": tgt_text,
        }


@torch.no_grad()
def greedy_decode(
    model,
    encoder_input: torch.Tensor,
    encoder_mask: torch.Tensor,
    target_tokenizer: Tokenizer,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    sos_id = target_tokenizer.token_to_id("[SOS]")
    eos_id = target_tokenizer.token_to_id("[EOS]")
    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.empty(1, 1, device=device, dtype=encoder_input.dtype).fill_(sos_id)
    while decoder_input.size(1) < max_length:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        logits = model.projection_layer.proj(out[:, -1])
        next_id = torch.argmax(logits, dim=-1)
        decoder_input = torch.cat([decoder_input, next_id.view(1, 1)], dim=1)
        if next_id.item() == eos_id:
            break
    return decoder_input.squeeze(0)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    loss_fn: nn.Module,
    target_tokenizer: Tokenizer,
    device: torch.device,
    max_length: int,
    writer: Optional[SummaryWriter],
    global_step: int,
    stage: str,
    decode_examples: int = 50,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_batches = 0.0, 0
    predicted: List[str] = []
    expected: List[str] = []
    for i, batch in enumerate(loader):
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        label = batch["label"].to(device)
        enc_out = model.encode(encoder_input, encoder_mask)
        dec_out = model.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
        logits = model.projection_layer.proj(dec_out)
        loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
        total_loss += float(loss.item())
        total_batches += 1
        if i < decode_examples and encoder_input.size(0) >= 1:
            out_ids = greedy_decode(model, encoder_input[0:1], encoder_mask[0:1], target_tokenizer, max_length, device)
            predicted.append(target_tokenizer.decode(out_ids.detach().cpu().numpy()))
            expected.append(batch["target_text"][0])
    avg_loss = total_loss / max(1, total_batches)
    metrics: Dict[str, float] = {f"{stage}_loss": avg_loss}
    if predicted:
        metrics[f"{stage}_cer"] = float(torchmetrics.CharErrorRate()(predicted, expected).item())
        metrics[f"{stage}_wer"] = float(torchmetrics.WordErrorRate()(predicted, expected).item())
        bleu = float("nan")
        try:
            sacre = torchmetrics.text.SacreBLEUScore()
            bleu = float(sacre(predicted, [[e] for e in expected]).item())
        except Exception:
            pass
        metrics[f"{stage}_bleu"] = bleu
    if writer is not None:
        writer.add_scalar(f"{stage}/loss", metrics[f"{stage}_loss"], global_step)
        if f"{stage}_cer" in metrics:
            writer.add_scalar(f"{stage}/cer", metrics[f"{stage}_cer"], global_step)
            writer.add_scalar(f"{stage}/wer", metrics[f"{stage}_wer"], global_step)
            writer.add_scalar(f"{stage}/bleu", metrics[f"{stage}_bleu"], global_step)
        writer.flush()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Code → Docstring (CodeSearchNet Python / drugi)")
    parser.add_argument("--dataset", default=DATASET_NAME, help="HF dataset (npr. code-search-net/code_search_net)")
    parser.add_argument("--dataset-config", default="", help="Konfiguracija dataseta ako postoji")
    parser.add_argument("--source-column", default=SOURCE_COLUMN, help="Kod: func_code_string ili code")
    parser.add_argument("--target-column", default=TARGET_COLUMN, help="Opis: func_documentation_string ili docstring")
    parser.add_argument("--language", default="python", help="Filter po jeziku (CodeSearchNet: python, go, ...); prazno = bez filtera")
    parser.add_argument("--run-dir", default="runs/code_comment")
    parser.add_argument("--force-rebuild-tokenizers", action="store_true")
    parser.add_argument("--epochs", type=int, default=5, help="Broj epoha (1 epoha = 1 prolaz kroz 100k, eval na kraju)")
    parser.add_argument("--batch-size", type=int, default=64, help="Veličina batch-a; evaluacija i ažuriranje težina na kraju svake epohe")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=100_000, help="Broj trening uzoraka (1 epoha = max_training batches)")
    parser.add_argument("--max-val", type=int, default=5_000)
    parser.add_argument("--max-test", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume-from", type=str, default=None, help="Putanja do checkpoint-a (npr. runs/code_comment/weights/epoch_003.pt) za nastavak treninga")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping (0 = isključeno)")
    args = parser.parse_args()
    ds_config = args.dataset_config.strip() or None

    cfg = RunConfig(
        dataset_name=args.dataset,
        dataset_config=ds_config,
        source_column=args.source_column,
        target_column=args.target_column,
        context_size=args.context_size,
        model_dimension=args.d_model,
        number_of_blocks=args.blocks,
        heads=args.heads,
        dropout=args.dropout,
        feed_forward_dimension=args.d_ff,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        max_train_samples=args.max_train,
        max_val_samples=args.max_val,
        max_test_samples=args.max_test,
    )
    run_dir = Path(args.run_dir)
    tokenizers_dir = ensure_dir(run_dir / "tokenizers")
    weights_dir = ensure_dir(run_dir / "weights")
    tb_dir = ensure_dir(run_dir / "tensorboard")
    save_json(run_dir / "run_config.json", asdict(cfg))
    set_seed(cfg.seed)

    language_filter = args.language.strip() or None
    print("Učitavam dataset", cfg.dataset_name, language_filter and f"(language={language_filter})" or "", "...")
    train_raw, val_raw, test_raw = load_hf_splits(
        cfg.dataset_name, cfg.dataset_config, cfg.seed,
        cfg.max_train_samples, cfg.max_val_samples, cfg.max_test_samples,
        language_filter=language_filter,
    )
    print("Train:", len(train_raw), "Val:", len(val_raw), "Test:", len(test_raw))

    src_tok_path = tokenizers_dir / "tokenizer_src.json"
    tgt_tok_path = tokenizers_dir / "tokenizer_tgt.json"
    src_tok = build_or_load_tokenizer(src_tok_path, train_raw, cfg.source_column, args.force_rebuild_tokenizers)
    tgt_tok = build_or_load_tokenizer(tgt_tok_path, train_raw, cfg.target_column, args.force_rebuild_tokenizers)

    train = filter_too_long(train_raw, src_tok, tgt_tok, cfg.source_column, cfg.target_column, cfg.context_size)
    val = filter_too_long(val_raw, src_tok, tgt_tok, cfg.source_column, cfg.target_column, cfg.context_size)
    test = filter_too_long(test_raw, src_tok, tgt_tok, cfg.source_column, cfg.target_column, cfg.context_size)
    print("Posle filter_too_long — Train:", len(train), "Val:", len(val), "Test:", len(test))

    train_ds = Seq2SeqDataset(train, src_tok, tgt_tok, cfg.source_column, cfg.target_column, cfg.context_size)
    val_ds = Seq2SeqDataset(val, src_tok, tgt_tok, cfg.source_column, cfg.target_column, cfg.context_size)
    test_ds = Seq2SeqDataset(test, src_tok, tgt_tok, cfg.source_column, cfg.target_column, cfg.context_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, 64), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(
        source_vocab_size=src_tok.get_vocab_size(),
        target_vocab_size=tgt_tok.get_vocab_size(),
        source_context_size=cfg.context_size,
        target_context_size=cfg.context_size,
        model_dimension=cfg.model_dimension,
        number_of_blocks=cfg.number_of_blocks,
        heads=cfg.heads,
        dropout=cfg.dropout,
        feed_forward_dimension=cfg.feed_forward_dimension,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-9)
    pad_id = tgt_tok.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1).to(device)
    writer = SummaryWriter(str(tb_dir))

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"Nastavljam trening od epohe {start_epoch} (global_step={global_step})")
    else:
        start_epoch = 0
        global_step = 0

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            enc_out = model.encode(encoder_input, encoder_mask)
            dec_out = model.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
            logits = model.projection_layer.proj(dec_out)
            loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = cfg.learning_rate * lr_scale
            optimizer.step()
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.flush()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        val_metrics = evaluate(
            model, val_loader, loss_fn, tgt_tok, device, cfg.context_size,
            writer, global_step, "val", decode_examples=50,
        )
        save_json(run_dir / "last_val_metrics.json", val_metrics)
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, weights_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "best_val_loss": best_val_loss,
            }, weights_dir / "best.pt")
        else:
            epochs_no_improve += 1
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping na epohi {epoch + 1}.")
            break

    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        bvl = ckpt.get("best_val_loss") or (ckpt.get("val_metrics") or {}).get("val_loss")
        print(f"Test na best.pt (epoch {ckpt['epoch']+1}, val_loss={bvl})")
    test_metrics = evaluate(
        model, test_loader, loss_fn, tgt_tok, device, cfg.context_size,
        writer=None, global_step=global_step, stage="test", decode_examples=200,
    )
    save_json(run_dir / "test_metrics.json", test_metrics)
    print("Gotovo.", run_dir, "Test:", test_metrics)
    print('Inferenca: python code_comment/infer_code_comment.py --run-dir "' + str(run_dir) + '" --code "def add(a, b): return a + b"')


if __name__ == "__main__":
    main()
