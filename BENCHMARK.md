# BENCHMARK — lora-lab

Résultats **réels** des runs LoRA effectués sur ma machine. Chaque run renvoie
au dossier `runs/<slug>/` qui contient `training.log`, `trainer_state.json`,
`adapter_model.safetensors` et `sample_generations.md`.

> Politique : aucun chiffre inventé. Si une métrique est absente, c'est parce
> qu'elle n'a pas été mesurée — pas extrapolée.

## Synthèse

| Run | Base | Target modules | seq_len | Loss step 100 | `train_loss` (mean) | Runtime | Adapter |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [#1](#run-1--qwen25-05b--lora-r8-attn-only) | Qwen2.5-0.5B (494 M) | q,v (2/L) | 512 | 1.824 | 2.125 | 528 s | 2.07 MB |
| [#2](#run-2--qwen25-05b--lora-r8-attn--mlp) | Qwen2.5-0.5B (494 M) | q,k,v,o,gate,up,down (7/L) | 256 | 1.683 | 1.818 | 325 s | 17.6 MB |
| [#3](#run-3--tinyllama-11b--lora-r8-attn-only) | TinyLlama-1.1B-Chat | q,v (2/L) | 512 | 1.286 | 1.420 | 257 s | 4.30 MB |

Conditions communes : Apple M1 Pro 16 GB, torch 2.8.0 MPS, fp32, dataset
`jpacifico/French-Alpaca-dataset-Instruct-55K` (subset 500), 100 steps,
batch=1 × grad_accum=4 (effective 4), lr=2e-4, warmup_ratio=0.03, seed=42.

---

## Run #1 — Qwen2.5-0.5B + LoRA r=8 (attn only)

| Champ | Valeur |
| --- | --- |
| **Run** | [`runs/qwen05b_lora_mps/`](runs/qwen05b_lora_mps) |
| **Config** | [`configs/qwen05b_lora_mps.yaml`](configs/qwen05b_lora_mps.yaml) |
| **target_modules** | `q_proj, v_proj` |
| **max_seq_length** | 512 |

### Loss (extrait de `trainer_state.json`, fenêtres de 10 steps)

| Step | Loss | Mean token acc |
| ---: | ---: | ---: |
| 10 | 2.888 | 0.538 |
| 30 | 2.224 | 0.580 |
| 50 | 1.947 | 0.625 |
| 70 | 1.849 | 0.630 |
| 100 | **1.824** | 0.649 |

`train_loss` (mean sur les 100 steps, reporté par HF) : **2.125**.

### Temps

| Métrique | Valeur |
| ---: | ---: |
| `train_runtime` | 528.03 s (≈ 8 min 48 s) |
| `train_samples_per_second` | 0.758 |
| `train_steps_per_second` | 0.189 |

---

## Run #2 — Qwen2.5-0.5B + LoRA r=8 (attn + MLP)

| Champ | Valeur |
| --- | --- |
| **Run** | [`runs/qwen05b_lora_mlp_mps/`](runs/qwen05b_lora_mlp_mps) |
| **Config** | [`configs/qwen05b_lora_mlp_mps.yaml`](configs/qwen05b_lora_mlp_mps.yaml) |
| **target_modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **max_seq_length** | **256** (rabaissé pour tenir en 16 GB MPS sans swap) |

### Loss

| Step | Loss | Mean token acc |
| ---: | ---: | ---: |
| 10 | 2.519 | 0.577 |
| 30 | 1.745 | 0.647 |
| 50 | 1.691 | 0.645 |
| 70 | 1.670 | 0.653 |
| 100 | **1.683** | 0.662 |

`train_loss` (mean) : **1.818**.

### Temps

| Métrique | Valeur |
| ---: | ---: |
| `train_runtime` | 325.2 s |
| `train_samples_per_second` | 1.230 |
| `train_steps_per_second` | 0.307 |

> **Caveat** : le runtime plus bas qu'au Run #1 vient surtout de
> `max_seq_length=256` (vs 512). À seq_len=512 sur 7 modules le throughput
> tombait à 30-60 s/step (swap MPS). Voir [`runs/qwen05b_lora_mlp_mps/COMPARISON.md`](runs/qwen05b_lora_mlp_mps/COMPARISON.md).

---

## Run #3 — TinyLlama-1.1B + LoRA r=8 (attn only)

| Champ | Valeur |
| --- | --- |
| **Run** | [`runs/tinyllama_lora_mps/`](runs/tinyllama_lora_mps) |
| **Config** | [`configs/tinyllama_lora_mps.yaml`](configs/tinyllama_lora_mps.yaml) |
| **target_modules** | `q_proj, v_proj` |
| **max_seq_length** | 512 |

### Loss

| Step | Loss | Mean token acc |
| ---: | ---: | ---: |
| 10 | 1.899 | 0.619 |
| 30 | 1.405 | 0.680 |
| 50 | 1.310 | 0.687 |
| 70 | 1.309 | 0.686 |
| 100 | **1.286** | 0.698 |

`train_loss` (mean) : **1.420**.

### Temps

| Métrique | Valeur |
| ---: | ---: |
| `train_runtime` | 257.3 s (≈ 4 min 17 s) |
| `train_samples_per_second` | 1.555 |
| `train_steps_per_second` | 0.389 |

> Curieux : TinyLlama 1.1B tourne 2× plus vite que Qwen 0.5B sur la même config.
> Probablement parce que l'archi Llama bénéficie de noyaux MPS plus optimisés
> que Qwen2 (à confirmer).

---

## Honest take

- Sur les 3 runs, **la loss baisse vraiment** (proof-of-life MPS, pas une convergence).
- L'écart final de loss entre les 3 n'est pas comparable directement : modèles
  différents, vocab différents → la cross-entropy absolue dépend du tokenizer.
- Pour des sorties utilisables il faudrait 1-2 epochs sur 10-50k exemples,
  soit ~30-60 h sur M1 Pro en fp32 — domaine GPU CUDA dédié à partir d'ici.
- Sample generations dans `runs/*/sample_generations.md` confirment qu'on apprend
  le format alpaca mais qu'on hallucine encore les faits (« capitale de l'Australie =
  Sydney » sur Qwen0.5B baseline, **Sydney aussi** sur TinyLlama+LoRA).

## Runs à venir

- [ ] Qwen2.5-1.5B — pas sûr que ça tienne en 16 GB MPS sans QLoRA
- [ ] Phi-3-mini (3.8 B) — OOM quasi certain en fp32 sur M1 Pro 16 GB
- [ ] Sweeps `r ∈ {4,8,16,32}` et `alpha ∈ {8,16,32,64}` — 8 runs supplémentaires
- [ ] QLoRA 4-bit (bnb) — bloqué jusqu'à accès CUDA

Voir [ROADMAP.md](https://github.com/Meuleur/AIproject/blob/main/ROADMAP.md).
