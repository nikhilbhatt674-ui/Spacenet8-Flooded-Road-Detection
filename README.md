# SpaceNet 8 Flooded-Road Detection

Research implementation for flooded-road segmentation using pre- and post-event satellite imagery from the **SpaceNet 8 Flood Detection Challenge**.

## Models

Four configurations were evaluated:

- Post-event U-Net
- Early-Fusion U-Net
- Siamese U-Net without difference fusion
- Full Siamese U-Net with pre/post feature comparison and road-prior branch

All models use a **ResNet-34 encoder**.

## Dataset

The experiments use the public **SpaceNet 8** training data from:

- Germany
- Louisiana-East

The SpaceNet 8 dataset is **not included in this repository** and must be downloaded separately from the official public distribution.

Road information is obtained from the SpaceNet 8 reference CSV files using:

- `ImageId`
- `Object == "Road"`
- `Wkt_Pix`
- `Flooded`

## Experimental Setup

- Input crop: 512 × 512
- Encoder: pretrained ResNet-34
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 8
- Maximum epochs: 10
- Random seed: 42
- Split seed: 42
- GPU used: NVIDIA Tesla T4

A geographic block split with buffering was used to reduce spatial leakage.

Final Protocol A split:

- Training: 312 tiles
- Validation: 101 tiles
- Test: 118 tiles

## Results

Validation-tuned test results:

| Model | Flooded-road F1 | Flooded-road IoU |
|---|---:|---:|
| Post-event U-Net | 0.077 | 0.040 |
| Early-Fusion U-Net | 0.214 | 0.120 |
| Siamese without difference | 0.269 | 0.156 |
| Full Siamese U-Net | **0.332** | **0.199** |

These results are from a **single-seed, 10-epoch experiment** and should be considered preliminary.

## Files

- `prepare_data.py` — prepares SpaceNet 8 imagery and road/flood labels
- `make_splits.py` — creates geographic train/validation/test splits
- `models.py` — model architectures
- `train.py` — model training and evaluation
- `aggregate.py` — aggregates experimental results
- `make_figures.py` — generates result figures
- `run_all.sh` — experiment runner
- `requirements.txt` — Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation and NVIDIA GPU are recommended.

## Example

Prepare the data:

```bash
python prepare_data.py --raw raw --out data/processed
```

Create the geographic split:

```bash
python make_splits.py \
  --manifest data/processed/manifest.csv \
  --out data/splits.csv \
  --split_seed 42
```

Example training run:

```bash
python train.py \
  --data data/processed \
  --splits data/splits.csv \
  --protocol A \
  --variant siam_full \
  --seed 42 \
  --epochs 10 \
  --out results/siam_full
```

## Reproducibility Note

The reported paper results use **Protocol A, seed 42, and a maximum of 10 epochs**. Cross-region Protocol B splits were prepared but were not executed for the reported study.

## Data Availability

SpaceNet 8 is publicly available through the SpaceNet Flood Detection Challenge. The dataset itself is not redistributed in this repository.

## Code Availability

The implementation used for this research is provided in this repository.
