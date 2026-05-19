# Recommandations pratiques — LoRA / QLoRA pour petits LM

Synthèse écrite après 3 vraies runs sur Apple M1 Pro 16 GB (voir
[BENCHMARK.md](BENCHMARK.md)). Pas de wisdom recopiée d'internet ici — seulement
ce qu'on a effectivement constaté.

## Quel hardware pour quel modèle

| Modèle | M1 Pro 16 GB (MPS, fp32) | Recommandation |
| --- | --- | --- |
| ≤ 1.1 B (TinyLlama, Qwen2.5-0.5B) | ✅ tient confortablement | Mac, fp32, LoRA attn-only |
| ~1.5 B (Qwen2.5-1.5B) | ⚠️ borderline, ~6 GB juste pour les poids fp32 | Mac OK mais réduire seq_len à 256 ; QLoRA-4bit recommandé sur CUDA |
| ≥ 3 B (Phi-3-mini, Mistral-7B) | ❌ OOM quasi certain | CUDA + QLoRA-4bit |

**Règle empirique constatée** : sur 16 GB unified memory, on tient un modèle
fp32 jusqu'à ~1.2 B params si on garde `max_seq_length ≤ 512` et qu'on cible
uniquement l'attention (2 modules/layer). Au-delà, on swap.

## Quels `target_modules` choisir

Constat (Run #1 vs Run #2 dans BENCHMARK) :

- **`q_proj, v_proj` (attention only, 2 modules/layer)** : c'est le défaut
  historique du papier LoRA. Adapter petit (2 MB sur Qwen 0.5B), peu de
  pression mémoire activations, suffisant pour apprendre un format / un style.
- **`q,k,v,o + gate,up,down` (attention + MLP, 7 modules/layer)** : ~8× la
  taille d'adapter, **~7-10× les params entraînables**, et sur MPS 16 GB ça
  oblige à diviser `max_seq_length` par 2 pour éviter le swap.
- **Pratique** : sur Mac, partir d'attn-only. Sur CUDA avec QLoRA-4bit, partir
  d'attn+MLP — c'est la config standard de PEFT/TRL.

## Quel `r` et `alpha`

Pas mesuré sur les sweeps complets encore (voir [#5](https://github.com/Meuleur/lora-lab/issues/5),
[#6](https://github.com/Meuleur/lora-lab/issues/6)), donc je m'en tiens à ce
qui est observé ailleurs et que nos runs confirment qualitativement :

- `r=8, alpha=16` (ratio 2:1) → bon défaut pour SFT, observe une loss qui
  baisse vite.
- Augmenter `r` aide surtout quand on attaque des tâches loin du pré-training
  (raisonnement, code structuré). Pour du simple alignment instruct sur
  Alpaca-FR : pas vu de gain notable au-delà de r=8 dans la littérature
  (Hu et al., 2021 ; Dettmers et al., 2023).
- `alpha` ≈ `r` ou `2×r`. Ratios extrêmes (alpha = 0.25×r ou 8×r) → plutôt
  pour fine-tuner des comportements subtils, ce qui dépasse notre use-case.

## Précision sur MPS

- **MPS ne supporte ni bf16 ni fp16 stables** au moins jusqu'à torch 2.8.0
  pour le training. Notre code force `bf16=fp16=False` quand `device=mps`
  (voir [`src/device.py`](src/device.py)).
- Conséquence : on tourne en fp32, donc ~2× plus de VRAM que sur CUDA bf16,
  et ~4× plus que sur CUDA QLoRA-4bit.
- Inférence MPS observée : **20-30 tok/s** stable sur les 3 modèles testés.

## Batch + grad accumulation

`per_device_train_batch_size=1, gradient_accumulation_steps=4` donne un batch
effectif de 4. Sur M1 Pro, augmenter le batch physique au-delà de 1 fait
swapper sur tous les modèles ≥ 0.5 B (testé). Augmenter `grad_accum` à 8 ou 16
reste libre mais on perd en wall-clock.

## Sauvegarde / commit des artefacts

L'adapter `safetensors` est petit (2-20 MB sur ces tailles de modèles),
**il peut être commité** dans le repo. Les choses à *ne pas* commiter :

- `checkpoint-*/` (optimizer state, ~10× la taille de l'adapter)
- `tokenizer.json` du base model (downloadable, dupliqué)
- `training_args.bin` (binaire pickle, le YAML de config est plus lisible)

Voir `.gitignore` pour la liste précise des allow-listes.

## Workflow recommandé sur cette machine

1. **Itérer sur les hyperparams en CPU** d'abord (`--dry-run` pour vérifier
   la config, `--max-steps 5` pour vérifier le pipeline).
2. **Run réel** : `python -m src.train --config configs/<run>.yaml --max-steps 100`
   pour un proof-of-life rapide (~5-10 min sur M1 Pro).
3. **Inspection qualitative** : `python -m src.generate --base-model ... --adapter runs/<run>/ --output runs/<run>/sample_generations.md`
4. **Merge + export** quand on est content : `python -m src.merge ...`
   puis `python -m src.gguf_export ...` pour Ollama / llama.cpp.

## Ce qu'on n'a PAS encore mesuré

À ne pas inventer : ces points restent ouverts (issues GitHub liées) :

- Effet réel de `r` au-delà de 8 (#5)
- Effet réel de `alpha` (#6)
- QLoRA-4bit comparaison (#4 — needs CUDA)
- Qualité sur des modèles plus gros (#2 Qwen1.5B, #3 Phi-3)

Tant que ces runs ne sont pas faits, **ne traitez pas ce README comme une
référence universelle**. Il documente ce qu'on a vu sur 3 runs spécifiques.
