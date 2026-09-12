# RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction

Official PyTorch implementation of **RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction**, accepted at **ECAI 2025**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)
[![PyTorch Geometric 2.6.1](https://img.shields.io/badge/PyG-2.6.1-34D058.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Usage & Training](#usage--training)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Evaluation & Logging](#evaluation--logging)
- [Troubleshooting](#troubleshooting)
- [Citation & Contact](#citation--contact)

---

## Overview

**RaMen** is a multi-strategy multi-modal recommendation framework designed for **Bundle Construction** (recommending/building sets of items for users). It integrates multi-modal item representation (content/visual and textual features) with hypergraph neural networks and asymmetric graph attention over item-user-item relationships.

Key architectural concepts implemented in RaMen:
- **Multimodal Feature Projection & Fusion**: Learns rich representations from visual features (`content_feature.pt`) and text descriptions (`description_feature.pt` / `openai_description_feature.pt`) using dedicated MLP projectors and a Self-Attention fusion module.
- **Asymmetric Item Module (GATv2 / Amatrix)**: Leverages Item-User-Item (IUI) graph topologies via Graph Attention networks (`AsymMatrix` / `Amatrix`) to capture complex item dependency patterns with residual connections.
- **Hypergraph Neural Network (HGNN)**: Models high-order relationships across items and bundles using learnable hyperedges with Gumbel-Softmax augmentation.
- **Contrastive Learning Losses**: Employs InfoNCE-style contrastive regularization at both item and bundle levels (`cl_loss_function`) alongside negative log-likelihood reconstruction loss (`recon_loss_function`).

---

## Key Features

- **Multi-Modal Learning**: Combines visual features, text embeddings, and collaborative filtering signals (`item_cf_feature.pt`).
- **Flexible Bundle Augmentation**: Supports online data augmentation strategies during training, including Item Dropout (`ID`) and Item Replacement (`IR`).
- **Hypergraph Modeling**: Learns modal-specific hyperedges (`v_hyper`, `t_hyper`) to aggregate bundle-item affinity across complex user behavior graphs.
- **Config-Driven Experiments**: Comprehensive configuration support (`config.yaml`) tailored for benchmark datasets (POG, Food, Electronic, MealRec, Spotify).
- **Metric Tracking & Checkpointing**: Automatic logging via Weights & Biases (WandB), TensorBoard, and checkpointing of best-performing models based on `Recall@20` and `NDCG@20`.

---

## System Architecture

The core model workflow consists of:

```
                  ┌─────────────────────────────────────┐
                  │ Multimodal Inputs (Visual & Text)   │
                  └──────────────────┬──────────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   │  MLP Projectors & Self-Attention  │
                   └─────────────────┬─────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│   Item-User-Item  │      │ Hypergraph Neural │      │ Transformer Bundle│
│   Asymmetric GAT  │      │ Network (HGNN)    │      │ Encoder           │
└─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
          │                          │                          │
          └──────────────────────────┼──────────────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │ Multimodal & Structural Fusion    │
                   └─────────────────┬─────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │ Reconstruction & Contrastive Loss │
                   └───────────────────────────────────┘
```

Detailed model code can be found in [models/RaMen.py](models/RaMen.py) and utility components in [models/utils.py](models/utils.py).

---

## Repository Structure

```
.
├── config.yaml          # Dataset-specific hyperparameters and configuration
├── train.py             # Main entry point for training and evaluating RaMen
├── utility.py           # PyTorch Dataset loaders and graph preprocessing utilities
├── requirements.txt     # Python dependency specifications
├── models/
│   ├── __init__.py      # Model exports (RAMEN)
│   ├── RaMen.py         # Main RAMEN architecture, HGNN, and Asymmetric GAT layers
│   └── utils.py         # TransformerEncoder, SelfAttention, and tensor utilities
└── README.md            # Repository documentation
```

---

## Prerequisites & Installation

### Requirements

- **Python**: `3.8+`
- **PyTorch**: `2.0.1` with CUDA 11.7 (`cu117`)
- **PyTorch Geometric**: `2.6.1`
- **Hardware**: NVIDIA GPU with CUDA support recommended

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hoang2306/RaMen.git
   cd RaMen
   ```

2. **Create and activate a Python virtual environment** (Conda or venv):
   ```bash
   conda create -n ramen python=3.10 -y
   conda activate ramen
   ```

3. **Install dependencies**:
   ```bash
   pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
   pip install -r requirements.txt
   ```

   *Dependencies included in `requirements.txt`:*
   - `numpy==1.24.4`
   - `PyYAML==6.0.2`
   - `scipy==1.14.1`
   - `torch_geometric==2.6.1`
   - `tqdm==4.66.5`
   - `wandb==0.18.2`

---

## Dataset Preparation

The training script expects dataset files to be located under `./datasets/<dataset_name>/`. Supported dataset keys configured in [config.yaml](config.yaml) include:
`pog`, `food`, `electronic`, `mealrec_H`, `mealrec_L`, `pog_dense`, `spotify`, `spotify_sparse`.

### Expected Directory Layout

```
datasets/
└── <dataset_name>/
    ├── count.json                   # JSON containing {"#U": num_users, "#B": num_bundles, "#I": num_items}
    ├── ui_full.txt                  # User-Item interactions (format: user_id, item_id1, item_id2...)
    ├── bi_train.txt                 # Bundle-Item training pairs (format: bundle_id, item_id1...)
    ├── bi_valid_input.txt           # Validation bundle-item input pairs
    ├── bi_valid_gt.txt              # Validation bundle-item ground-truth target pairs
    ├── bi_test_input.txt            # Test bundle-item input pairs
    ├── bi_test_gt.txt               # Test bundle-item ground-truth target pairs
    ├── content_feature.pt           # PyTorch tensor for visual/content embeddings
    ├── description_feature.pt       # PyTorch tensor for text description embeddings
    ├── item_cf_feature.pt          # PyTorch tensor for collaborative filtering item features
    └── <iui_path>.npy               # NumPy array containing IUI graph edge indices (for GATv2)
```

---

## Configuration

Dataset-level default configurations are declared in [config.yaml](config.yaml). Example configuration block for `spotify`:

```yaml
spotify:
  data_path: './datasets'
  batch_size_train: 256
  batch_size_test: 256
  topk: [10, 20, 40, 80]
  neg_num: 1
  embedding_sizes: [64]
  num_layerss: [1]
  lrs: [1.0e-3]
  l2_regs: [1.0e-5]
  epochs: 100
  test_interval: 1
```

Command-line parameters passed to [train.py](train.py) automatically override these default YAML settings.

---

## Usage & Training

### Basic Training Command

To train **RaMen** on the default dataset (`spotify`):

```bash
python train.py --dataset spotify --model RAMEN --gpu 0
```

### Example Commands

- **Train on `food` dataset with GPU 0**:
  ```bash
  python train.py --dataset food --model RAMEN --gpu 0 --lr 0.0001
  ```

- **Train on `pog` dataset with WandB logging enabled**:
  ```bash
  python train.py --dataset pog --model RAMEN --gpu 0 --wandb 1
  ```

- **Custom Hyperparameters (Bundle/Item Loss & Hyperedges)**:
  ```bash
  python train.py --dataset spotify --model RAMEN --gpu 0 \
      --hyper_num 4 \
      --bundle_alpha 1.0 \
      --item_alpha 1.0 \
      --alpha_bundle_loss 0.05 \
      --alpha_item_loss 0.1 \
      --epochs 100
  ```

---

## Hyperparameter Reference

`train.py` accepts the following command-line flags:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-g`, `--gpu` | `str` | `"0"` | Target GPU device index |
| `-d`, `--dataset` | `str` | `"spotify"` | Dataset name (must match `config.yaml`) |
| `-m`, `--model` | `str` | `"CLHE"` | Model name registered in `models` (e.g. `RAMEN`) |
| `-l`, `--lr` | `float` | `1e-4` | Learning rate for Adam optimizer |
| `-r`, `--reg` | `float` | `1e-5` | L2 regularization weight (weight decay) |
| `--bundle_alpha` | `float` | `1.0` | Weight hyperparameter for bundle fusion |
| `--item_alpha` | `float` | `1.0` | Weight hyperparameter for item fusion |
| `--alpha_residual` | `float` | `1.0` | Weight hyperparameter for residual connections |
| `--hyper_num` | `int` | `4` | Number of hyperedges in hypergraph module |
| `--n_layer_gat` | `int` | `1` | Number of GATv2 layers in asymmetric item module |
| `--alpha_bundle_loss`| `float` | `0.05` | Weight for bundle contrastive loss |
| `--alpha_item_loss`  | `float` | `0.1` | Weight for item contrastive loss |
| `--residual_type`   | `int` | `0` | Type of residual connection (`0` standard, `1` scaled) |
| `--bundle_augment`   | `str` | `"ID"` | Bundle augmentation strategy: `ID` (Dropout) or `IR` (Replacement) |
| `--bundle_ratio`     | `float` | `0.5` | Ratio of reserved items in bundle augmentation |
| `--num_token`        | `int` | `200` | Maximum token sequence length per bundle |
| `--seed`             | `int` | `2306` | Random seed for reproducible training |
| `--wandb`            | `int` | `0` | Set to `1` to enable Weights & Biases logging |
| `--iui_path`         | `str` | `""` | Filename path (without `.npy`) for IUI graph edge index |
| `--custom_log_path`  | `str` | `""` | Custom path prefix for experiment log file |

---

## Evaluation & Logging

During training, `train.py` automatically evaluates the model on validation and test sets every `test_interval` epoch based on `topk` settings (e.g. `@5`, `@10`, `@20`, `@40`, `@80`).

### Metrics Evaluated

- **`Recall@K`**: Proportion of ground-truth bundle items retrieved in top-K predictions.
- **`NDCG@K`**: Normalized Discounted Cumulative Gain accounting for rank order.

### Output Artifacts

Running an experiment generates log files and model checkpoints in the following directories:
- **`./log/<dataset>/<model>/`**: Text log file containing epoch metrics and configuration summary.
- **`./runs/<dataset>/<model>/`**: TensorBoard event logs.
- **`./checkpoints/<dataset>/<model>/model`**: Model PyTorch state dictionary saved when `Recall@20` and `NDCG@20` improve.
- **`./checkpoints/<dataset>/<model>/conf`**: Dumped JSON configuration of the best run.

---

## Troubleshooting

- **Missing Dataset Features Error**:
  If `content_feature.pt` or `description_feature.pt` are missing from `./datasets/<dataset_name>/`, `utility.py` will print `[ERROR] no content_feature & description_feature` and set modal features to `None`. Ensure features are extracted and placed in the dataset folder.
- **CUDA Device Out of Memory**:
  Adjust `batch_size_train` in [config.yaml](config.yaml) or pass a lower batch size for dense datasets (`pog_dense`, `spotify`).
- **`iui_path` npy file missing**:
  Pass `--iui_path <filename>` specifying the `.npy` edge index file located inside your dataset directory.

---

## Citation & Contact

If you find **RaMen** useful for your research, please cite our ECAI 2025 paper:

```bibtex
@inproceedings{nguyen2025ramen,
  title={RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction},
  author={Nguyen, Huy-Son and Nguyen, Quang-Huy and Pham, Duc-Hoang and Le, Duc-Trong and Le, Hoang-Quynh and Sitkrongwong, Padipat and Takasu, Atsuhiro and Mansoury, Masoud},
  booktitle={ECAI 2025},
  pages={1889--1896},
  year={2025},
  url={https://arxiv.org/abs/2507.14361}
}
```

For questions or issues, please open an issue in this repository.

---

## License

This repository is distributed under the terms of the MIT License.
