# REMDO: Residual Equilibrium Manifold Learning for Multidisciplinary Design Optimization

REMDO is a framework to learn residual manifolds of coupled multidisciplinary systems that can be used for downstream system-level decision-making including surrogate modeling, uncertainty quantification, and design optimization. Currently, REMDO is compatible with OpenMDAO, where it creates _residual taps_ which are then used to generate data for the equlibrium manifold learning.

# Overview
REMDO trains PyTorch/BoTorch surrogate models on residual evaluations from coupled multidisciplinary systems. The current workflow is:

1. Define or wrap a coupled OpenMDAO system.
2. Evaluate residual taps across the input and coupling-variable space.
3. Train multitask Gaussian-process surrogates for the residual equations.
4. Use active learning to adaptively sample the residual manifold.
5. Reuse the learned residual equilibrium manifold for downstream analysis.

# Installation
Install the package in editable mode from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

# Repository layout
- `src/remdo/`: reusable REMDO package code.
- `examples/`: example scripts and notebooks.
- `tests/`: test suite location.

# Runtime configuration
REMDO uses a central runtime configuration for PyTorch device placement and floating-point precision:

```python
import torch
from remdo import configure

configure(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
```

# Examples
Example experiment scripts run serially on a laptop and distribute repetitions across MPI ranks when launched with MPI:

```bash
python examples/satellite_active_learning.py --reps 4
mpiexec -n 4 python examples/satellite_active_learning.py --reps 20
```

Each repetition uses a distinct randomized initial GP design seed.

# Turbine FEM requirements
Running the turbine test function example requires [MATLAB Runtime R2024b](https://www.mathworks.com/products/compiler/matlab-runtime.html) to be installed and added to the system path.
