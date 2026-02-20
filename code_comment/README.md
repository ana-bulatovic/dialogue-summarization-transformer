# Code → Docstring (opis koda)

Projekat za **generisanje opisa koda u prirodnom jeziku**: ulaz je Python kod (npr. funkcija), izlaz je kratak docstring (šta funkcija radi).

**Primer:**  
`def add(a, b): return a + b` → *"Add two numbers."* / *"Function that sums two numbers."*

## Skup podataka

Korišćen je HuggingFace dataset **[CM/codexglue_code2text_python](https://huggingface.co/datasets/CM/codexglue_code2text_python)** (CodeXGlue Code-to-Text, Python):

- ~280k parova (kod, docstring)
- Kolone: `code` (telо funkcije), `docstring` (opis u prirodnom jeziku)
- Train/validation/test splitovi

Ako neki drugi dataset ima drugačije nazive kolona, trening možeš prilagoditi argumentima:

- `--source-column` (npr. `code`)
- `--target-column` (npr. `docstring`)

## Metode

- **Model:** encoder–decoder Transformer iz korena projekta (`model.py`), isti kao za sažimanje dijaloga (SAMSum).
- **Tokenizacija:** WordLevel (whitespace) za kod i za docstring; posebni tokeni `[SOS]`, `[EOS]`, `[PAD]`, `[UNK]`.
- **Trening:** cross-entropy na decoder izlazu, Adam, LR warmup, gradient clipping, early stopping, čuvanje `best.pt` po validacionom loss-u.

## Kako pokrenuti

Iz korena repozitorijuma (folder `transformer`):

**Trening (podrazumevano: 20k train, 1k val/test, 10 epoha):**
```bash
python code_comment/train_code_comment.py --run-dir runs/code_comment
```

Opcije: `--max-train`, `--epochs`, `--batch-size`, `--context-size`, `--d-model`, itd.

**Inferenca (opis za zadati kod):**
```bash
python code_comment/infer_code_comment.py --run-dir runs/code_comment --code "def add(a, b): return a + b"
```

Izlaz: generisani docstring (opis funkcije).

## Struktura run foldera

- `runs/code_comment/run_config.json` — konfiguracija treninga  
- `runs/code_comment/tokenizers/` — tokenizer_src.json, tokenizer_tgt.json  
- `runs/code_comment/weights/` — best.pt, epoch_*.pt  
- `runs/code_comment/tensorboard/` — logovi za TensorBoard  
- `runs/code_comment/test_metrics.json` — test loss, CER, WER, BLEU  

Zavisnosti: `torch`, `datasets`, `tokenizers`, `torchmetrics`, `tqdm`, `tensorboard`; root projekta sadrži `model.py` i `dataset.py` (causal_mask).
