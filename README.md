# Deformable Object Shaping

Class project for Applied Optimal Control (w' 23). Code is closely based on code from
[DiffTaichi](https://github.com/taichi-dev/difftaichi)
and [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab), but rewritten in PyTorch and choosing components from each for our new task (deformable object shaping by kinematic control of a rigid body).

## Installation

Requires torch 2.0, tqdm, and [mmint_utils](https://github.com/MMintLab/mmint_utils).
Then run `pip install -e .` from the root directory.

## Basic Usage

To run an example trajectory optimization loop, run the following:

```
python scripts/diffmpm_torch_batch.py
```