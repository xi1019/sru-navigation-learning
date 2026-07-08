# Path-conditioned SRU Navigation Learning - RL Training Framework

[Paper](https://arxiv.org/abs/2506.05997) | [Website](https://michaelfyang.github.io/sru-project-website/) *(base SRU)*
Paper: TODO | [Website](https://leggedrobotics.github.io/rl-path-following/) *(path conditioning, this fork)*

**📌 Important Note**: This repository contains the RL training framework for the SRU navigation project, providing neural network architectures and on-policy training algorithms (PPO/MDPO). This repository does not include the simulation environments or task definitions. See the [original SRU project](https://michaelfyang.github.io/sru-project-website/) for the complete base navigation system, or the [path-conditioned project](https://leggedrobotics.github.io/rl-path-following/) for the path-conditioning extension added in this fork.

**🔀 Fork Notice**: This repository is a fork of [leggedrobotics/sru-navigation-learning](https://github.com/leggedrobotics/sru-navigation-learning) (Robotic Systems Lab, ETH Zurich). It adds **path conditioning** on top of the original SRU architecture: `ActorCriticSRU` can additionally be conditioned on a waypoint path, encoded by a new `PathEncoder` module and fused into the actor/critic features after the SRU. This is not part of the original paper or upstream repository — see [Path Conditioning (fork addition)](#path-conditioning-fork-addition) below, and the accompanying [project page](https://leggedrobotics.github.io/rl-path-following/) ("Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation", IROS 2026), for details.

## Overview

End-to-end RL training framework for visual navigation with SRU (Spatially-Enhanced Recurrent Units) architecture. This repository extends the original [rsl_rl](https://github.com/leggedrobotics/rsl_rl) framework with:

- **SRU Architecture**: Advanced recurrent networks with spatial transformation operations for implicit spatial memory
- **Attention Mechanisms**: Self-attention and cross-attention modules for multimodal fusion
- **Training Algorithms**: PPO, SPO, and MDPO (with Deep Mutual Learning) implementations
- **Multi-Camera Support**: Native handling of 1-2 camera inputs with proper masking

This framework is designed to train navigation policies that achieve 23.5% improvement over standard RNNs and enable zero-shot sim-to-real transfer.

## What's Included

✅ **Neural Network Architectures**
- ActorCriticSRU: SRU-based policy network with attention mechanisms
- LSTM_SRU: Core recurrent unit with spatial transformation gates
- Cross-attention fusion module for vision + proprioception
- 3D positional encoding for volumetric features
- PathEncoder: waypoint-path self-attention + cross-attention pooling module (**fork addition**, see below)

✅ **RL Training Algorithms**
- PPO (Proximal Policy Optimization)
- SPO (Symmetric Policy Optimization)
- MDPO (Multi-Distillation Policy Optimization with Deep Mutual Learning)
- MUON optimizer support for stable training with reduced memory usage (newly added, not in original paper)

✅ **Training Infrastructure**
- GPU-accelerated training pipeline
- Rollout storage with hidden state tracking
- Temporally consistent dropout across trajectories
- Multi-sensor observation handling

✅ **Model Export**
- JIT export for PyTorch deployment
- ONNX export for cross-platform inference (C++, ROS, TensorRT)
- Explicit hidden state I/O for recurrent models in ONNX format

✅ **Logging & Monitoring**
- Tensorboard integration
- Weights & Biases support
- Neptune logging

## What's NOT Included

❌ Simulation environments (Isaac Lab tasks, maze & path generation, terrain)
❌ Task definitions (observation spaces, reward functions, action interfaces)
❌ Robot models and locomotion policies
❌ Depth encoder pretraining infrastructure

**Note**: The simulation environment must be installed separately. See [sru-navigation-sim](https://github.com/mateo-haro/path-conditioned-sim) for the IsaacLab extension.

## Architecture

The repository is organized as follows:

```
path-conditioned-rl-nav/
├── config/                    # Training configuration files
│   └── dummy_config.yaml      # Example PPO configuration
├── rsl_rl/                    # Main Python package
│   ├── algorithms/            # RL algorithms
│   │   ├── ppo.py             # Proximal Policy Optimization
│   │   ├── spo.py             # Symmetric Policy Optimization
│   │   └── mdpo.py            # Multi-Distillation Policy Optimization
│   ├── modules/               # Neural network architectures
│   │   ├── actor_critic.py              # Basic MLP actor-critic
│   │   ├── actor_critic_recurrent.py    # RNN-based actor-critic
│   │   ├── actor_critic_sru.py          # SRU architecture (primary)
│   │   └── normalizer.py                # Observation normalization
│   ├── networks/              # Network components
│   │   └── sru_memory/        # SRU memory modules
│   │       ├── lstm_sru.py              # LSTM with SRU gating
│   │       ├── attention.py             # Cross-attention fusion with 3D positional encoding
│   │       └── path_encoder.py          # Waypoint-path encoder (fork addition)
│   ├── runners/               # Training orchestration
│   │   └── on_policy_runner.py          # Main training loop
│   ├── storage/               # Experience replay
│   │   └── rollout_storage.py           # Trajectory buffer
│   ├── env/                   # Environment interface
│   │   └── vec_env.py                   # Vectorized environment wrapper
│   └── utils/                 # Utilities
│       ├── trajectory_handler.py        # Padding/unpadding helpers
│       └── logging.py                   # Logging utilities
├── licenses/                  # Dependency licenses
├── setup.py                   # Package installation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

### Key Components

**ActorCriticSRU** ([rsl_rl/modules/actor_critic_sru.py](rsl_rl/modules/actor_critic_sru.py))
- Multi-input architecture: depth images + proprioceptive information + waypoint path (**fork addition**)
- Processing pipeline: Self-attention → Cross-attention → SRU → MLP, with the path latent fused in after the SRU, right before the MLP head
- Separate actor and critic networks with shared depth encoder
- Time embedding for critic value estimation
- Supports 1-2 camera inputs with automatic padding/masking

**LSTM_SRU** ([rsl_rl/networks/sru_memory/lstm_sru.py](rsl_rl/networks/sru_memory/lstm_sru.py))
- Multi-layer LSTM with SRU-style spatial transformation gates
- Polynomial refinement for forget gate (from research paper)
- Element-wise transformation operations for spatial memory
- Orthogonal weight initialization

**CrossAttentionFuseModule** ([rsl_rl/networks/sru_memory/attention.py](rsl_rl/networks/sru_memory/attention.py))
- Fuses volumetric (image) features with proprioceptive state
- Self-attention → Feed-forward → Cross-attention architecture
- 3D positional encoding for spatial awareness
- Efficient batched attention outside RNN loop

**PathEncoder** ([rsl_rl/networks/sru_memory/path_encoder.py](rsl_rl/networks/sru_memory/path_encoder.py)) — **fork addition**
- Encodes a fixed-length waypoint path (`num_waypoints` × `path_input_dim`, e.g. 15 waypoints of `[x, y, z, distance]`) into a compact latent
- Sinusoidal positional encoding → self-attention/feed-forward block → learned-query cross-attention pooling
- Computed in batch mode over all timesteps and concatenated onto the SRU output (bypasses the recurrent memory itself)
- Separate instances for actor (`attn_path_net`) and critic (`attn_critic_path_net`), mirroring the existing image-attention split

**Training Algorithms** ([rsl_rl/algorithms/](rsl_rl/algorithms/))
- PPO: Standard clipped policy optimization with value loss
- MDPO: Deep Mutual Learning with KL-divergence distillation between two networks
- Adaptive and fixed learning rate schedules

## Path Conditioning (fork addition)

This fork adds waypoint-path conditioning to `ActorCriticSRU`, so the policy can use an
externally provided route rather than relying solely on egocentric image/proprioceptive input.

**New constructor arguments** on `ActorCriticSRU`:
- `path_length` (default `60`): flattened length of the path observation (`num_waypoints * path_input_dim`)
- `path_input_dim` (default `4`): raw features per waypoint, e.g. `[x, y, z, distance]`
- `path_embed_dim` (default `32`): latent dimension produced by `PathEncoder`

**Observation layout**: the path segment must sit immediately before the image block, i.e.
`[proprioceptive, path, image(s)]` for the actor and `[proprioceptive, path, height, image(s), time]`
for the critic — mirroring where image/height blocks already live at the end of the observation
vector.

**Data flow**: unlike the image/height features (fused before the SRU via cross-attention), the
path latent bypasses the SRU entirely and is concatenated directly onto its output, right before
the final actor/critic MLP head. This keeps the recurrent memory focused on egocentric history,
since the path is assumed static for the episode and needs no temporal aggregation of its own.

JIT (`export_jit`) and ONNX (`export_onnx`) export both include the path encoder in the exported
graph, for all camera configurations (1 or 2 cameras).

## Installation

### Standalone Installation

```bash
git clone https://github.com/mateo-haro/path-conditioned-rl-nav.git
cd path-conditioned-rl-nav

python3 -m venv sru_nav_learning
source sru_nav_learning/bin/activate
pip install -e .
```

### Installation with Isaac Lab

If you're using this package with Isaac Lab, you need to replace the pre-installed `rsl_rl` package.

#### Directory Structure

You can place this repository anywhere, but the recommended structure is:

```
IsaacLab/
├── source/
│   ├── isaaclab/              # Core Isaac Lab
│   ├── isaaclab_assets/       # Asset library
│   ├── isaaclab_rl/           # RL framework wrappers
│   ├── isaaclab_tasks/        # Task definitions
│   └── isaaclab_nav_task/     # Your navigation task extension
├── rsl_rl/                    # This SRU-enhanced RL framework (recommended location)
├── _isaac_sim/                # Isaac Sim installation
└── isaaclab.sh                # Isaac Lab launcher script
```

**Alternative locations**:
- Inside `source/` directory (e.g., `source/rsl_rl/`)
- Standalone location outside IsaacLab
- Any custom location (adjust paths accordingly)

#### Installation Steps

```bash
# 1. Navigate to your IsaacLab installation
cd /path/to/IsaacLab

# 2. Uninstall the pre-installed rsl_rl from Isaac Lab
./isaaclab.sh -p -m pip uninstall rsl-rl-lib -y

# 3. Remove any cached rsl_rl directory (if exists)
rm -rf _isaac_sim/kit/python/lib/python3.10/site-packages/rsl_rl

# 4. Clone or place this repository
# Option A: Clone at IsaacLab root level (recommended)
git clone https://github.com/mateo-haro/path-conditioned-rl-nav.git rsl_rl

# Option B: Clone in source/ directory
cd source
git clone https://github.com/mateo-haro/path-conditioned-rl-nav.git rsl_rl
cd ..

# 5. Install this SRU-enhanced version in editable mode
cd rsl_rl  # Adjust path if you placed it elsewhere
../isaaclab.sh -p -m pip install -e .

# 6. Verify installation
../isaaclab.sh -p -c "from rsl_rl.modules import ActorCriticSRU; print('✓ SRU modules loaded')"
```

**Important Notes**:
- The package installs as `rsl_rl` (not `rsl_rl_lib`) to maintain compatibility with Isaac Lab imports
- This repository directory can have any name (e.g., `path-conditioned-rl-nav`, `rsl_rl`, etc.), but the installed package name will always be `rsl_rl`
- The editable install (`-e`) allows you to modify the code without reinstalling

**Dependencies**:
- PyTorch (GPU-accelerated training recommended)
- NumPy
- Optional: tensorboard, wandb, neptune (for logging)

**Note**: This package provides only the RL training framework. To train navigation policies, you also need:
1. A compatible simulation environment (e.g., [sru-navigation-sim](https://github.com/leggedrobotics/sru-navigation-sim))
2. A pretrained depth encoder (see project website)

## Usage

### Configuration

Training hyperparameters are specified in YAML configuration files. See [config/dummy_config.yaml](config/dummy_config.yaml) for an example PPO configuration.

Key parameters:
- `policy`: Network architecture configuration (SRU layers, attention heads, hidden dimensions)
- `algorithm`: PPO/MDPO hyperparameters (learning rate, entropy coefficient, clip range)
- `runner`: Training settings (number of steps per rollout, max iterations)

### Training

The typical training workflow with a compatible environment:

```python
from rsl_rl.runners import OnPolicyRunner
from your_env import YourNavigationEnv  # From simulation package

# Initialize environment
env = YourNavigationEnv(num_envs=4096)

# Create runner with configuration
runner = OnPolicyRunner(env, config, device='cuda:0')

# Train
runner.learn(num_learning_iterations=10000)
```

### Logging

The framework supports multiple logging backends configured through the `logger` parameter:
- **Tensorboard**: https://www.tensorflow.org/tensorboard/
- **Weights & Biases**: https://wandb.ai/site
- **Neptune**: https://docs.neptune.ai/

### Model Export

Export trained policies for deployment:

```python
# Load trained model
policy = ActorCriticSRU(...)
policy.load_state_dict(torch.load("checkpoint.pt"))

# Export to JIT (PyTorch deployment)
policy.export_jit(path="./exported", filename="policy.pt", normalizer=obs_normalizer)

# Export to ONNX (C++, ROS, TensorRT deployment)
policy.export_onnx(path="./exported", filename="policy.onnx", normalizer=obs_normalizer)
```

**ONNX Export Notes for Recurrent Models**:
- Hidden states are exposed as explicit inputs/outputs (`h_in`, `c_in` → `h_out`, `c_out`)
- Initialize hidden states to zeros at episode start
- Pass updated hidden states back as inputs for the next timestep

## Key Features

### SRU-Specific Enhancements

1. **Spatial Transformation Gates**: Element-wise multiplication operations enabling implicit spatial memory from egocentric observations

2. **Multi-Camera Fusion**: Native support for multiple depth cameras with proper padding, masking, and attention mechanisms

3. **Temporally Consistent Dropout**: Dropout masks maintained across trajectory timesteps for stable training

4. **Efficient Attention**: Batched self-attention and cross-attention computed outside RNN loop for computational efficiency

5. **Deep Mutual Learning**: MDPO algorithm with KL-divergence distillation between dual networks

6. **MUON Optimizer Integration**: Momentum Orthogonalized by Newton-schulz optimizer for hidden weight layers
   - Uses Newton-Schulz iteration for efficient orthogonalization in bfloat16
   - Reduces memory usage compared to Adam optimizer
   - Provides more stable training dynamics for deep networks
   - Automatically separates hidden weights (optimized with MUON) from biases/gains (optimized with AdamW)

7. **Path Conditioning** (newly added, not in original paper): A `PathEncoder` compresses an
   externally provided waypoint path into a latent that's fused into the actor/critic features
   after the SRU — see [Path Conditioning (fork addition)](#path-conditioning-fork-addition)

### Performance

- **23.5%** improvement over standard RNNs (LSTM/GRU)
- **29.6%** advantage vs. explicit mapping approaches
- **105%** better than stacked-frame baselines
- **2.5x** improvement in challenging stair environments
- Zero-shot sim-to-real transfer with no fine-tuning

## Related Projects

| Repository | Description |
|------------|-------------|
| [sru-pytorch-spatial-learning](https://github.com/leggedrobotics/sru-pytorch-spatial-learning) | Core SRU PyTorch module (standalone) |
| [sru-navigation-sim](https://github.com/leggedrobotics/sru-navigation-sim) | IsaacLab simulation environments |
| [sru-depth-pretraining](https://github.com/leggedrobotics/sru-depth-pretraining) | Self-supervised depth encoder pretraining |
| [sru-robot-deployment](https://github.com/leggedrobotics/sru-robot-deployment) | Real robot deployment (ROS2, Gazebo) |

## Citation

If you use the base SRU architecture in your research, please cite:

```bibtex
@article{yang2025sru,
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  title = {Spatially-enhanced recurrent memory for long-range mapless navigation via end-to-end reinforcement learning},
  journal = {The International Journal of Robotics Research},
  year = {2025},
  doi = {10.1177/02783649251401926},
  url = {https://doi.org/10.1177/02783649251401926}
}
```

If you use the path-conditioning extension introduced in this fork (see
[Path Conditioning (fork addition)](#path-conditioning-fork-addition)), please also cite:

```bibtex
@inproceedings{haro2026pathconditioned,
  title     = {Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation},
  author    = {Haro, Mateo and Richter, Julia and Yang, Fan and Cadena, Cesar and Hutter, Marco},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
}
```

See the [project page](https://leggedrobotics.github.io/rl-path-following/) for details on the
path-conditioning approach.

## Contribution Guidelines

For documentation, we adopt the [Google Style Guide](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings.

We use the following tools for maintaining code quality:

- [pre-commit](https://pre-commit.com/): Runs formatters and linters over the codebase
- [black](https://black.readthedocs.io/en/stable/): Code formatter
- [flake8](https://flake8.pycqa.org/en/latest/): Style checker

To set up pre-commit hooks:

```bash
# Installation (one time)
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Credits

This repository is a fork of [sru-navigation-learning](https://github.com/leggedrobotics/sru-navigation-learning)
(Robotic Systems Lab, ETH Zurich), which is itself built upon
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) from ETH Zurich Robotic Systems Lab and NVIDIA.

**Original rsl_rl Maintainers**: David Hoeller and Nikita Rudin
**SRU Extension**: Fan Yang
**Path Conditioning (this fork)**: Mateo Haro
**Affiliation**: Robotic Systems Lab, ETH Zurich
**Contact**: maharo@ethz.ch

## License

This project is licensed under the **BSD-3-Clause License** - see the [LICENSE](LICENSE) file for details.

See [licenses/](licenses/) directory for dependency licenses.
