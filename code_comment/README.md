# Code → Docstring (opis koda)

Projekat za **generisanje opisa koda u prirodnom jeziku**: ulaz je Python kod (npr. funkcija), izlaz je kratak docstring (šta funkcija radi).

**Primer:**  
`def add(a, b): return a + b` → *"Add two numbers."* / *"Function that sums two numbers."*

## Skup podataka

**Podrazumevano:** HuggingFace **[code-search-net/code_search_net](https://huggingface.co/datasets/code-search-net/code_search_net)** (CodeSearchNet), samo Python:

- Kolone: `func_code_string` (kod funkcije), `func_documentation_string` (docstring)
- Splitovi: train / valid / test; filter `--language python`
- Trening: 100k uzoraka, batch 64, 5 epoha; na kraju svake epohe računa se evaluacija i čuvaju težine

Alternativa: `--dataset CM/codexglue_code2text_python --source-column code --target-column docstring`.

Prilagodbe: `--source-column`, `--target-column`, `--language` (prazno = bez filtera).

## Metode

- **Model:** encoder–decoder Transformer iz korena projekta (`model.py`), isti kao za sažimanje dijaloga (SAMSum).
- **Tokenizacija:** WordLevel (whitespace) za kod i za docstring; posebni tokeni `[SOS]`, `[EOS]`, `[PAD]`, `[UNK]`.
- **Trening:** cross-entropy na decoder izlazu, Adam, LR warmup, gradient clipping, early stopping, čuvanje `best.pt` po validacionom loss-u.

## Kako pokrenuti

Iz korena repozitorijuma (folder `transformer`):

**Trening (CodeSearchNet Python: 100k train, batch 64, 5 epoha, evaluacija na kraju svake epohe):**
```bash
python code_comment/train_code_comment.py --run-dir runs/code_comment
```

**Nastavak treninga od prethodne epohe (resume):**
```bash
python code_comment/train_code_comment.py --run-dir runs/code_comment --resume-from runs/code_comment/weights/epoch_003.pt
```

Opcije: `--max-train`, `--epochs`, `--batch-size`, `--context-size`, `--d-model`, `--resume-from`, itd.

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
