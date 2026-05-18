# lora-lab

Banc d'essai pour fine-tuner de petits LM (TinyLlama, Qwen2.5-0.5B/1.5B, Phi-3-mini)
en **LoRA** et **QLoRA**, comparer les hyperparams (`r`, `alpha`, modules ciblés), et
exporter les modèles fusionnés en GGUF.

## Objectifs

- Pipeline réutilisable de fine-tuning LoRA avec `trl.SFTTrainer`
- Sweeps `r ∈ {4, 8, 16, 32}`, `alpha ∈ {8, 16, 32, 64}`
- Comparaison LoRA / QLoRA (bnb-4bit) sur 4 modèles
- Merge + export GGUF (llama.cpp) pour inférence locale
- Benchmark honnête (qualité, VRAM, vitesse)

## Structure

```
lora-lab/
├── README.md
├── requirements.txt
├── configs/          # configs YAML/JSON par run
├── src/              # loaders, trainer, merge utils
├── tests/            # tests pytest (sans GPU)
├── runs/             # outputs d'entraînement (ignorés par git)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

Les tests ne demandent pas de GPU : ils valident la config, le loader et la
logique de merge avec des mocks. Les runs de fine-tuning eux-mêmes nécessitent
une GPU CUDA (bnb-4bit n'est pas dispo sur Apple Silicon).

Part of [Meuleur/AIproject](https://github.com/Meuleur/AIproject).
