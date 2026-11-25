# Mean-Fluctuating Decomposition Network

**English** | [中文](README.md)

**Particle displacement and foot-contact force prediction for liquid–rigid interaction**

A point-cloud graph neural network surrogate for dense particle–rigid interaction: given the current state of 20,000 SPH particles and the 1–5 s motion sequence of a quadruped leg (or an excavator bucket), the network directly regresses the particle displacement field — decomposed into a mean flow plus a structural residual — together with the six-dimensional contact force/torque on the four feet (single-foot and four-foot output heads).

> Each particle sample is organized as 400 clusters × 50 neighbors (20,000 = 400 × 50). The rigid geometry is resampled to 110 points, and 729 virtual boundary points are added as environmental constraints.

\---

## 1\. Repository structure

```
.
├── train.py                                  # Training entry point (TensorBoard + cosine annealing + force metrics)
├── Visualization.py                          # PyVista 3×2 comparison: ground truth vs. error field for 1 s / 3 s / 5 s
├── models/
│   └── Mean-Fluctuating Decomposition.py     # Network definition (GeometryEncoder / StructuralResidualGNN / AttentionForceHead)
├── data\\\_utils/
│   ├── 读取npy\\\_data.py                       # Training / test Dataset (.npy format)
│   └── Data\\\_Visualization.py                 # Visualization Dataset (data/20000/\\\*.npz)
├── data/20000/
│   ├── 1s|2s|3s|4s|5s/
│   │   ├── par/      # particle states, \\\*.npz
│   │   └── rigid/    # rigid-body geometry sequences, \\\*.npz
├── rigid/single-leg/                         # 1.csv joint trajectory, q1.stl/q2.stl meshes, center-of-mass offsets
├── tools/check\\\_dataset.py                    # Dataset self-check (shapes / keys / swapped directories)
├── result/ournew/check/best\\\_model.pth        # Pretrained weights (used by the visualization)
└── requirements.txt
```

\---

## 2\. Installation

```bash
# Python 3.8+; install the PyTorch build that matches your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

`Visualization.py` additionally depends on `torch\\\_cluster` (PyG ecosystem, must match your torch/CUDA versions):

```bash
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

\---

## 3\. Data format

Run `python tools/check\\\_dataset.py` to validate every `.npz` file (keys, shapes, frame counts) before training or visualization.

### 3.1 Particle states — `data/20000/<duration>/par/<id>.npz`

|Key|Shape|Unit|Description|
|-|-|-|-|
|`current\\\_xyz`|(400, 50, 3)|mm|Particle coordinates at the current step (400 clusters × 50 neighbors)|
|`next\\\_xyz`|(400, 50, 3)|mm|Coordinates at the end of the sequence (supervision target)|
|`chazhi`|(400, 50, 3)|mm|`next\\\_xyz - current\\\_xyz`, total displacement|
|`mean\\\_xyz`|(400, 3)|mm|Mean-flow component (one vector per cluster)|
|`res`|(400, 50, 3)|mm|Residual relative to the mean flow, `chazhi = mean\\\_xyz + res`|
|`voxel\\\_xyz`|(400, 50, 3)|–|Initial velocity feature (already normalized)|
|`centroid\\\_xyz`|(400, 3)|mm|Cluster-center coordinates (voxel-cell center, not the cluster mean)|

All coordinates are divided by `160` at load time for non-dimensionalization (see `data\\\_utils/`).

### 3.2 Rigid-body geometry sequences — `data/20000/<duration>/rigid/<id>.npz`

|Key|Shape|Description|
|-|-|-|
|`geometry`|(T, 8, 110)|8 channels: `x, y, z, vx, vy, vz, gravity, dry`; 110 resampled geometry points; coordinates already divided by 160|

`T` depends on the duration: `1s→34`, `2s→67`, `3s→100`, `4s→134`, `5s→167` (0.03 s per frame).
`gravity` and `dry` (wetness) are environmental scalar channels; the network normalizes gravity by `/9.81` internally.

### 3.3 Boundary points

Generated analytically in `data\\\_utils/\\\*.py::generate\\\_boundary\\\_points()`: 729 points × 6 dims (position + zero velocity), acting as the interaction boundary between the rigid body and the fluid.

\---

## 4\. Quick start (visualization)

```bash
python Visualization.py
```

The script loads `result/ournew/check/best\\\_model.pth` and performs step-by-step rollouts for `TARGET\\\_INDEX=6` of `data/20000/` at 1 s / 3 s / 5 s, showing

* left column: ground-truth particle position field (colored by displacement magnitude)
* right column: model prediction error field

Press **S** in the window to export `Publication\\\_Ready\\\_Figure\\\_MultiStep.png`; the `Global vmin/vmax` sliders keep the color scale of both columns synchronized.

\---

## 5\. Training

```bash
python train.py --model "Mean-Fluctuating Decomposition" \\\\
                --batch\\\_size 32 --epoch 300 --learning\\\_rate 1e-4 --gpu 0
```

Logs and checkpoints are written to `./log/part\\\_seg/<timestamp>/{logs,check}`.

> \\\*\\\*Note\\\*\\\*: `train.py` expects the full training set (`./data/train`, `./data/test`, `.npy` format). This repository ships only 16 example samples (`data/20000/\\\*.npz`), which are sufficient to reproduce the visualization but \\\*\\\*not\\\*\\\* to train the model. The dataset generation scripts and force labels (`li.csv`) are not included; please contact the authors for the full dataset.

\---

## 6\. Results

|Setting|Weights|Notes|
|-|-|-|
|5 s|`result/ournew/check/best\\\_model.pth`|167-frame rollout|

The quantitative results reported in the paper (MSE / MRE, force MAE / NRMSE) are printed by the evaluation branch of `train.py`; all figures are produced by `Visualization.py`.

\---

\---

## 7\. Citation

If you use this code, please cite (to be completed):

```bibtex
@article{xxx2025mean,
  title   = {Mean-Fluctuating Decomposition Network for Liquid--Rigid Interaction},
  author  = {XXX},
  journal = {XXX},
  year    = {2025}
}
```

## 8\. License

MIT License — see [LICENSE](LICENSE).

