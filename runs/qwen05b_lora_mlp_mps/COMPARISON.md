# Attention only vs Attention + MLP — Qwen2.5-0.5B

Comparaison de deux configs LoRA sur le même base model + même dataset,
en n'écrasant **que** la liste `target_modules`.

| Métrique | `qwen05b_lora_mps` (attn only) | `qwen05b_lora_mlp_mps` (attn + MLP) |
| --- | --- | --- |
| `target_modules` | `q_proj, v_proj` (2/layer) | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (7/layer) |
| `max_seq_length` | 512 | **256** (rabaissé pour tenir en 16 GB MPS) |
| Steps | 100 | 100 |
| Train loss (step 10 → final) | 2.888 → **2.125** | 2.946 → **1.818** |
| Train runtime | 528 s | 325 s |
| Adapter size | 2.07 MB | **17.6 MB** (8.5×) |
| Trainable LoRA params (approx) | ~700 k | ~6 M |

## Lecture honnête

La loss finale est nettement plus basse avec MLP cible (1.82 vs 2.13). **Mais
ce n'est pas une comparaison parfaite** : on a réduit `max_seq_length` de 512
à 256 pour la run MLP, sinon la run swap MPS et plafonne à 60+ s/step sur
16 GB unified memory. Conséquence : par sample, le MLP voit moitié moins de
tokens → la baisse de loss est sur-estimée.

À retenir tout de même :
- **Plus de modules = plus de capacité d'adaptation**, mais 8× la taille
  d'adapter et 8× plus de pression mémoire activations.
- Sur Apple Silicon 16 GB, viser **attention only** par défaut, n'élargir aux
  MLP que si on peut réduire `max_seq_length` ou si on a une vraie GPU CUDA
  avec plus de VRAM.
- Sur CUDA avec QLoRA-4bit, élargir aux MLP devient quasi gratuit en VRAM
  → c'est la config standard recommandée par Hugging Face / PEFT.
