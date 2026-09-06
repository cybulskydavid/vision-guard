# VisionGuard

**A research platform for measuring how easily modern visual search engines can be fooled — and how much of that weakness comes from the way we compress images into numbers.**

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792EE5?logo=lightning&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/domain-adversarial%20robustness-critical)
![Task](https://img.shields.io/badge/task-image%20retrieval-informational)

---

## The problem, in one minute

When you search Google Images with a photo, when a phone unlocks with a face, when a shop finds "this exact sneaker", or when a CCTV system re-identifies a person across cameras — none of those systems compare pixels. They turn every image into a short list of numbers, an **embedding**, and then look for the nearest neighbours in that number space.

That design is what makes visual search fast and flexible. It is also its blind spot.

An attacker can add a pattern of noise to a photo that a human eye cannot see — the picture still looks identical — while the embedding quietly slides across the number space and lands next to the wrong neighbours. The retrieval system does not crash. It does not warn anyone. It confidently returns the wrong results.

This is a harder failure to catch than a mislabelled classifier. A classifier says "cat" and you can check it against "dog". A retrieval system just hands back a ranked list, and nobody can tell by looking whether it was manipulated.

**VisionGuard exists to put numbers on that risk.** It is a controlled laboratory that answers questions such as:

- How much invisible noise does it actually take to break image retrieval built on today's foundation models (CLIP, DINO)?
- Are large vision transformers more or less fragile than convolutional networks?
- We routinely squeeze 768-dimensional embeddings down to 256 to save storage and speed up search — **does that compression make the system easier to attack?**
- Does *how* you learn that compression (trained metric learning vs. plain statistical PCA) change how robust the final system is?
- Are these results real, or seed luck? (Every experiment is repeated across multiple random seeds.)

---

## What this repository actually is

Not a single script. A **modular experiment bench** where every part of the pipeline is a swappable component, so a researcher can run a full combinatorial study without rewriting code.

|  Dial you can turn        | Options available today                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| **Backbone**              | CLIP ViT-B/16, CLIP ResNet-50, DINO ViT-B/16, DINO ViT-S/16, DINO ResNet-50 |
| **Dataset**               | CUB-200-2011 (birds), Stanford Cars, Stanford Online Products (22 634 classes) |
| **Embedding head**        | Raw backbone features · PCA projection · trained projection (Triplet or ArcFace) |
| **Attack**                | FGSM (one shot) · PGD (iterative, multi-restart) · Carlini–Wagner (optimisation-based) |
| **Attack budget**         | Any ε (e.g. 1/255 … 8/255) or C&W margin κ                                    |
| **Repetitions**           | Independent seeds for *model training* and for *the attack itself*            |
| **Reported metrics**      | Recall@1 and mAP, clean vs. attacked, mean ± standard deviation across seeds |

Multiplied out, that is thousands of legitimate experiment configurations — all driven by the same reusable core in `src/`.

---

## Inside `src/` — the engine room

This is where the value of the repository sits. `scripts/` only chooses *which* experiment to run; `src/` is what makes any experiment possible. Roughly 1 100 lines of dependency-light, single-responsibility Python.

```
src/
├── schemas/       what to run on   → declarative model & dataset definitions
├── data/          what goes in     → class-disjoint dataset splits with identity tracking
├── modules/       what is measured → the embedding heads under test
├── attacks/       the adversary    → FGSM, PGD, Carlini–Wagner in embedding space
├── lightning/     the harness      → runners, data plumbing, measurement callbacks
└── utils/         the plumbing     → model zoo, mining, caching, paths, logging
```

### `src/schemas/` — the control panel

Two tiny files that keep the whole study honest.

- **`models.py`** — every backbone declared as a name plus its output width (512, 768, 1024, 2048). Scripts never hard-code a dimension; they read it from the schema, so a model swap is a one-word change.
- **`datasets.py`** — every dataset declared with its path and, crucially, its **class-disjoint splits**: the classes used for training, validation and testing never overlap (e.g. Cars: classes 0–69 train, 70–97 validation, 98–195 test).

> **Why a reviewer should care:** class-disjoint splitting is the difference between a benchmark and a self-congratulation. The model is always tested on object categories it has *never* seen, which is exactly how retrieval systems are used in the real world.

### `src/data/` — datasets that remember who is who

**`class_subset.py`** wraps any standard image folder dataset and filters it down to an allowed set of classes, remapping labels to a clean, contiguous range. Each item it returns carries four things: the image, its label, **its index**, and **its filename**.

Those last two fields are not decoration. The index is what lets the attack look up the right neighbours in the gallery and what lets evaluation exclude an image from being its own search result. The filename makes every single adversarial example traceable back to the original photo on disk.

### `src/modules/` — the compression layer under the microscope

- **`l2_norm_layer.py`** — projects embeddings onto the unit sphere, so "similar" always means cosine similarity and nothing drifts because of vector length.
- **`projection_layer_wrapper.py`** — bolts a small trainable head onto a **frozen** backbone, with the normalisation applied on both sides. It also overrides `train()` so the frozen backbone can never be silently flipped back into training mode by the framework — a classic, hard-to-spot source of invalid results.

> This is the heart of the research question: the expensive foundation model stays untouched, and only the cheap compression layer changes. That isolates the variable being studied *and* makes the whole study affordable, because no experiment ever retrains a giant network.

### `src/attacks/` — the adversary

The most substantial part of the codebase, and the part that is genuinely non-standard.

Off-the-shelf adversarial attacks assume a classifier: they push an image away from label A towards label B. **Retrieval has no labels at inference time.** So every attack here was re-derived to work on *geometry* — the attack succeeds when the image's embedding drifts away from things that genuinely match it and towards things that do not.

| File | What it implements | Character |
| ---- | ------------------ | --------- |
| `base.py` | Abstract `BaseAttack` interface — `execute()` plus a `margin` property | Contract that makes attacks interchangeable; adding a fourth attack means one new file |
| `fgsm.py` | Fast Gradient Sign Method | One gradient step. Cheap, fast, the baseline "how fragile is this at all?" probe |
| `pgd.py` | Projected Gradient Descent | Iterative, with random restarts and strict ε-ball projection; keeps the best adversarial example found per image |
| `cw.py`  | Carlini–Wagner | The strong one: tanh-space reparameterisation, **binary search over the trade-off constant**, per-image tracking of the smallest successful perturbation, and a best-effort fallback for images it cannot break |

All three share the same recipe: mine each image's nearest same-class ("positive") and nearest different-class ("negative") neighbours, then optimise a triplet-style objective on cosine similarity, keeping the result inside a valid image range. FGSM and PGD are budget-constrained (*"given ε of invisible noise, how much damage?"*); C&W is the mirror image (*"what is the smallest noise that guarantees damage?"*).

The `@ensure_eval` decorator guarantees the model is in evaluation mode for the duration of an attack and restored afterwards — no accidental dropout or batch-norm updates corrupting an attack run.

### `src/lightning/` — the experiment harness

**`modules/`** — three thin runners, one per kind of experiment:

- `evaluation_module.py` — clean baseline: encode, normalise, report.
- `adversarial_module.py` — attack each batch, then encode the attacked images. It also keeps the perturbed images themselves for C&W, so perturbations can be inspected and visualised.
- `pca_evaluation_module.py` — extracts features across a dataset, computes PCA, then **converts the PCA basis into a frozen `nn.Linear` layer** and saves it. That single trick means the statistical baseline and the trained head become the exact same object type, so the attacks and evaluation code cannot tell them apart — a genuinely fair comparison.
- `projection_layer_module.py` — the training loop for the projection head: Triplet or ArcFace loss, optional miner, AdamW, cosine-annealed learning rate, and optimisation restricted to the head and loss parameters only.

**`datamodules/class_data_module.py`** — one place that defines how data is loaded (batch size, workers, pinned memory, prefetching, optional class-balanced sampler), so every experiment gets identical data handling.

**`callbacks/`** — the measurement instruments, and the most quietly rigorous code in the repo:

- `leave_one_out_metrics_callback.py` — computes Recall@1 and mAP where **every image in turn is the query and all the others are the gallery**, with the query itself masked out so it can never retrieve itself and score a free point. Similarity is computed in **chunks** with explicit GPU cache clearing, which is what allows evaluation on Stanford Online Products (tens of thousands of classes) without exhausting memory. It also persists the embedding gallery to disk for downstream reuse.
- `adv_leave_one_out_metrics_callback.py` — the attacked variant: adversarial queries are scored against the **clean** gallery, which is exactly the real-world threat model (an attacker manipulates the query they submit, not the database they are searching). It masks by image identity, not just position.
- `leave_one_out_validation_metrics_callback.py` — a lightweight version of the same idea used during training, so checkpoints are selected on retrieval quality (`val_mAP`) rather than on training loss.

### `src/utils/` — the plumbing that makes long runs survivable

| File | What it does | Why it matters |
| ---- | ------------ | -------------- |
| `model_loader.py` | Loads any supported backbone behind **one function call**, and returns the matching preprocessing pipeline split into an image stage and a normalisation stage | CLIP and DINO expect different normalisation. The split is deliberate: attacks must operate in real `[0, 1]` pixel space, with normalisation applied *inside* the model — otherwise the "invisible" noise budget is meaningless |
| `mining.py` | Finds each query's nearest positives and negatives in the gallery, with the query itself excluded | The targeting system for every attack, written with aggressive tensor cleanup to survive large galleries |
| `cache_tools.py` | Detects the last completed batch on disk | A C&W run over a big dataset takes many hours; this is what makes it **resumable after a crash** instead of restarted from zero |
| `path_creator.py` | Builds a deterministic result path from model, dataset, dimension, attack, iterations, margin and seed | Results are self-describing — a folder path *is* the experiment configuration, so nothing gets mixed up across thousands of runs |
| `decorators.py` | `@ensure_eval` — forces and restores evaluation mode | Cheap insurance against a whole class of silent methodological bugs |
| `logger.py` | Consistent console logging | Long unattended runs stay auditable |

---

## How one experiment flows

```
  images ──▶ ClassSubset ──▶ frozen backbone ──▶ L2 norm ──▶ projection head ──▶ L2 norm ──▶ embedding
   (CUB /                    (CLIP / DINO)                    (PCA or trained)                  │
    Cars /                                                                                      │
    SOP)                                                                                        ▼
                                                                                     ┌──────────────────┐
        ┌──────────────────────────────────────────────────────────────────────┐     │  clean gallery   │
        │  attack (FGSM / PGD / C&W)  ◀── mining: nearest positives & negatives │◀────│   (saved .pt)    │
        └───────────────────────────────┬──────────────────────────────────────┘     └──────────────────┘
                                        ▼                                                       │
                            adversarial images ──▶ same model ──▶ adversarial embeddings        │
                                                                            │                   │
                                                                            ▼                   ▼
                                                         leave-one-out retrieval scoring ◀──────┘
                                                                            │
                                                                            ▼
                                                      Recall@1 / mAP · clean vs. attacked · mean ± std
```

---

## Engineering decisions worth calling out

- **Frozen backbones by design.** Only a small head is ever trained. A study that would otherwise need weeks of GPU time fits into a research schedule, and the comparison stays clean because the expensive part of the model is byte-identical across every condition.
- **Attacks operate on geometry, not labels.** Standard attack libraries do not cover retrieval; these were adapted from first principles to attack embedding neighbourhoods.
- **Correct leave-one-out evaluation.** The query is always removed from its own gallery — by identity, not just by position. This is a widely mishandled detail that inflates published retrieval numbers.
- **Memory-safe at scale.** Chunked similarity computation plus explicit tensor deletion lets the same code run on a 6 000-image dataset and on Stanford Online Products.
- **Crash-resilient.** Every batch is cached to disk; interrupted runs resume at the right batch, and caches are cleaned up on successful completion.
- **Two independent sources of randomness.** The *model seed* (which trained head) and the *attack seed* (the randomness inside the attack) are separated, so it is always clear whether a result is a property of the model or an artefact of one attack roll.
- **Statistics, not anecdotes.** Configurations are trained and evaluated across five seeds, with margin selection done on validation data and results reported as mean ± standard deviation.
- **Extensible on purpose.** New attack → implement `BaseAttack`. New backbone → one schema entry plus one branch in the loader. New dataset → one schema entry. Nothing else changes.

---

## `scripts/` — experiments already wired up

| Script | What it runs |
| ------ | ------------ |
| `evaluation/base_evaluation.py` | Clean retrieval baseline for raw backbone features, across all models and datasets |
| `evaluation/pca_evaluation.py` | Fits and saves a PCA projection, then evaluates the compressed embeddings |
| `evaluation/train_and_evaluate_metric_learning.py` | Full study: margin selection on validation, multi-seed training with Triplet/ArcFace, then test evaluation with aggregated statistics |
| `evaluation/eval_ml.py` | CLI re-evaluation of already-trained heads (`--models`, `--datasets`, `--loss-types`, `--load-mode`), discovering saved configurations from disk |
| `evaluation/evaluate_metrics_learning.py` | Re-scores a previously saved results bundle |
| `adversarial/adversarial_attack.py` | Attacks raw backbone embeddings across the ε grid |
| `adversarial/pca_adversarial_attack.py` | Attacks PCA-compressed embeddings |
| `adversarial/adv_ml.py` | Attacks trained metric-learning embeddings, automatically selecting the best training margin from validation scores on disk |

---

## Reading the results

| Term | Plain meaning |
| ---- | ------------- |
| **Embedding** | The short list of numbers that represents an image. Similar images sit close together. |
| **Recall@1** | How often the single most similar image returned is genuinely the same object. Higher is better. |
| **mAP** | How good the *whole* ranked list is, not just the top hit. Higher is better. |
| **ε (epsilon)** | The attacker's noise budget. `8/255` means every pixel may move by at most 8 steps out of 256 — invisible to the eye. |
| **κ (kappa)** | For Carlini–Wagner: how decisively the attack must win before it stops shrinking the noise. |
| **Clean vs. attacked** | The same metric before and after the attack. The gap between them *is* the result. |

A typical headline from this platform reads: *"under an invisible ε = 4/255 perturbation, Recall@1 on this backbone falls from X to Y, and compressing embeddings from 768 to 256 dimensions changes that drop by Z."*

---

## Running it

**Requirements:** Python 3.12+, a CUDA GPU (CPU works but is impractical for iterative attacks).

```bash
pip install -r requirements.txt
```

DINO ResNet-50 is pulled from `torch.hub` on first use, and CLIP/DINO transformer weights are downloaded on first use too, so the first run of a new backbone needs network access.

**Point it at your data.** Dataset roots and class splits live in `src/schemas/datasets.py`; each dataset is expected in standard `ImageFolder` layout (one directory per class).

**Run an experiment** from the repository root, so that `src` resolves as a package:

```bash
python -m scripts.evaluation.base_evaluation                    # clean baseline
python -m scripts.evaluation.train_and_evaluate_metric_learning # train + evaluate projection heads
python -m scripts.adversarial.adv_ml                            # attack the trained system
```

Which models, datasets, losses, budgets and seeds run is controlled by the loops at the bottom of each script. Outputs land in a self-describing tree:

```
features/<loss>/<attack>/<iters>/<margin>/<model>/<dataset>/<dim>/<seed>/features.pt
metric_learning/projection_layers/<loss>/<model>/<dataset>/<dim>d_margin<m>_seed<s>.pt
logs/<loss>/<model>/<dataset>/margin<m>_seed<s>/
cache/...        # per-batch resume cache, cleared automatically on completion
```

---

## Repository map

```
vision-guard/
├── src/
│   ├── attacks/
│   │   ├── base.py                                    # BaseAttack interface
│   │   ├── fgsm.py                                    # single-step attack
│   │   ├── pgd.py                                     # iterative attack with restarts
│   │   └── cw.py                                      # Carlini–Wagner with binary search
│   ├── data/
│   │   └── class_subset.py                            # class-filtered dataset with index + filename
│   ├── lightning/
│   │   ├── callbacks/
│   │   │   ├── leave_one_out_metrics_callback.py      # chunked clean retrieval metrics
│   │   │   ├── adv_leave_one_out_metrics_callback.py  # adversarial queries vs. clean gallery
│   │   │   └── leave_one_out_validation_metrics_callback.py
│   │   ├── datamodules/
│   │   │   └── class_data_module.py                   # unified dataloader configuration
│   │   └── modules/
│   │       ├── evaluation_module.py                   # clean inference
│   │       ├── adversarial_module.py                  # attack + inference
│   │       ├── pca_evaluation_module.py               # PCA fitted and frozen into a linear layer
│   │       └── projection_layer_module.py             # metric-learning training loop
│   ├── modules/
│   │   ├── l2_norm_layer.py                           # unit-sphere normalisation
│   │   └── projection_layer_wrapper.py                # frozen backbone + trainable head
│   ├── schemas/
│   │   ├── models.py                                  # backbone registry
│   │   └── datasets.py                                # dataset paths + class-disjoint splits
│   └── utils/
│       ├── model_loader.py                            # one-call model + transform loading
│       ├── mining.py                                  # positive / negative neighbour mining
│       ├── cache_tools.py                             # resume support
│       ├── path_creator.py                            # deterministic result paths
│       ├── decorators.py                              # @ensure_eval
│       └── logger.py                                  # consistent logging
├── scripts/
│   ├── evaluation/                                    # clean, PCA and metric-learning studies
│   └── adversarial/                                   # attack studies
├── requirements.txt                                   # pinned-floor runtime dependencies
├── CITATION.cff                                       # how to cite this work
├── LICENSE                                            # MIT
└── .gitignore                                         # excludes galleries, caches, checkpoints, datasets
```

All experiment output (`features/`, `cache/`, `logs/`, `metric_learning/`, `pca/`, `*.pt`, `*.ckpt`) is deliberately kept out of version control — it is large, fully regenerable and machine-specific. Datasets are never committed either.

---

## Notes for reviewers

This is a research codebase, written to support a study rather than to ship as a product. A few things follow from that, stated plainly:

- Experiment sweeps are configured in the `if __name__ == "__main__"` block of each script rather than through a config file; `eval_ml.py` is the exception and takes command-line arguments.
- Dataset locations are absolute paths in `src/schemas/datasets.py` and need to be adjusted for a new machine.
- Some log messages are in Polish, reflecting the academic context the project was developed in.

Everything that carries scientific weight — the splits, the mining, the attacks, the metric computation — lives in `src/` and is reusable, tested by repeated multi-seed runs, and deliberately kept independent of any single experiment.

---

## Citing this work

If this platform or its results are useful in your own research, please cite it. GitHub renders the metadata in [`CITATION.cff`](CITATION.cff) as a **"Cite this repository"** button in the sidebar, which will generate a BibTeX or APA entry for you.

## License

Released under the [MIT License](LICENSE) — free to use, modify and build on, commercially or otherwise, with attribution.

The pretrained backbones it loads carry their own terms: **CLIP** (MIT, OpenAI) and **DINO** (Apache 2.0, Meta AI). The datasets referenced in `src/schemas/datasets.py` — CUB-200-2011, Stanford Cars and Stanford Online Products — are distributed by their respective authors under their own licenses and are not included in this repository.
