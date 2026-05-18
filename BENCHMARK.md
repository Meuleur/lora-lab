# BENCHMARK — lora-lab

Résultats **réels** des runs LoRA réalisés sur ma machine. Chaque entrée renvoie
au dossier `runs/<slug>/` qui contient le `training.log`, le `trainer_state.json`,
les générations exemples et l'adapter `safetensors`.

> Politique : aucun chiffre inventé. Si une métrique est absente, c'est parce
> qu'elle n'a pas été mesurée — pas parce qu'elle a été extrapolée.

---

## Run #1 — Qwen2.5-0.5B + LoRA r=8 (MPS)

| Champ | Valeur |
| --- | --- |
| **Run** | `runs/qwen05b_lora_mps/` |
| **Base model** | `Qwen/Qwen2.5-0.5B` (494 M params, float32) |
| **Adapter** | LoRA r=8, alpha=16, dropout=0.05, target `q_proj, v_proj`, bias none |
| **Dataset** | `jpacifico/French-Alpaca-dataset-Instruct-55K`, **subset 500 lignes**, `max_seq_length=512` |
| **Hardware** | Apple M1 Pro, 16 GB RAM, backend MPS (torch 2.8.0) |
| **Précision** | float32 (bf16/fp16 désactivés sur MPS — pas stables) |
| **Quantization** | aucune (bitsandbytes = CUDA only) |
| **Batch effectif** | 4 (`per_device_train_batch_size=1` × `gradient_accumulation_steps=4`) |
| **Steps** | 100 (cap via `--max-steps 100`) |
| **LR** | 2.0e-4, `warmup_ratio=0.03`, scheduler linear par défaut |
| **Seed** | 42 |

### Temps & throughput

| Métrique | Valeur |
| --- | --- |
| `train_runtime` | **528.03 s** (≈ 8 min 48 s) |
| `train_samples_per_second` | 0.758 |
| `train_steps_per_second` | 0.189 |
| `total_flos` | 1.13 × 10¹⁴ |

Throughput observé : ~3 à 13 s/step selon la longueur de la séquence (`packing=False`).

### Loss (extrait de `trainer_state.json`)

| Step | Loss | Mean token accuracy |
| --- | --- | --- |
| 10 | 2.888 | 0.538 |
| 20 | 2.524 | 0.567 |
| 30 | 2.223 | 0.580 |
| 40 | 2.276 | 0.580 |
| 100 (final, moyenné) | **2.125** | — |

La loss baisse nettement sur les 30 premiers steps puis plafonne — comportement
attendu pour 500 exemples + 100 steps : c'est un **proof-of-life**, pas une
convergence sérieuse.

### Taille adapter sur disque

| Fichier | Taille |
| --- | --- |
| `adapter_model.safetensors` | 2 175 168 octets (**2.07 MB**) |
| `adapter_config.json` | 1 018 octets |

L'adapter est commité tel quel dans le repo. Les dossiers `checkpoint-50/` et
`checkpoint-100/` (≈ 17 MB chacun, contiennent l'optimizer state et le
tokenizer downloadé en double) sont **ignorés** par `.gitignore` — pour
reproduire un checkpoint complet, relancer `python -m src.train`.

### Générations qualitatives

5 prompts FR fixes — voir `runs/qwen05b_lora_mps/sample_generations.md`.
Throughput inférence MPS observé : **25-29 tok/s** sur prompts moyens
(0.5-1 s/token au démarrage à cause du warmup MPS, puis stable).

### Honest take

100 steps × 500 exemples = ~80 % d'epoch. Le modèle apprend le format alpaca
(« ### Instruction / ### Response ») mais répète beaucoup et hallucine sur les
faits (« capitale de l'Australie = Sydney »). Pour des sorties utilisables il
faudrait au minimum 1-2 epochs complètes sur 10-50k exemples, ce qui sur M1 Pro
prendrait ~30-60 h en float32 — domaine du GPU dédié à partir d'ici.

---

## Runs à venir (skipped — nécessitent CUDA)

- Run LoRA TinyLlama-1.1B / Qwen2.5-1.5B / Phi-3-mini
- Variantes QLoRA (bnb-4bit)
- Sweeps `r ∈ {4,8,16,32}` et `alpha ∈ {8,16,32,64}`
- LoRA *attention only* vs *attention + MLP*

Voir `ROADMAP.md` du repo `Meuleur/AIproject` pour le suivi.
