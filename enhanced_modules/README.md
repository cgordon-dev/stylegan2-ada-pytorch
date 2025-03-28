# Enhanced StyleGAN2-ADA Workload Modules

This directory contains enhanced tools and utilities for StyleGAN2-ADA workloads, building on the base workload modules. Additions include:

1. **Unified CLI Interface**: One command-line tool to access all workloads
2. **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
3. **Unit Tests**: Test suite for workload modules
4. **Docker Support**: Containerized environment for reproducible runs

## Unified CLI Interface

The `stylegan_cli.py` script provides a single entry point to all workload modules:

```bash
# Training
python stylegan_cli.py train --dataset-path ./datasets/automotive --image-size 256 --outdir ./results

# Image generation
python stylegan_cli.py generate --network ./results/model.pkl --seeds 0-10 --outdir ./results/generated

# Fine-tuning
python stylegan_cli.py fine-tune --dataset-path ./datasets/medical --resume ffhq256 --outdir ./results

# Latent vector optimization
python stylegan_cli.py optimize-latent --network ./results/model.pkl --target ./targets/target.png --outdir ./results/optimized

# Mixed precision training
python stylegan_cli.py mixed-precision --dataset-path ./datasets/fashion --mixed-precision-mode aggressive --outdir ./results
```

## Experiment Tracking

All workloads support integration with TensorBoard and Weights & Biases:

```bash
# Use TensorBoard for tracking
python stylegan_cli.py train --dataset-path ./datasets/automotive --tensorboard --log-dir ./logs

# Use Weights & Biases for tracking
python stylegan_cli.py train --dataset-path ./datasets/automotive --wandb --wandb-project my-project --wandb-entity my-team
```

To view TensorBoard logs:

```bash
tensorboard --logdir=./logs
```

## Unit Tests

The `tests` directory contains unit tests for all workload modules:

```bash
# Run all tests
cd enhanced_modules
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_workloads
```

## Docker Support

Docker configuration is provided for containerized execution:

```bash
# Build and run training container
cd enhanced_modules
docker-compose build stylegan-train
docker-compose run stylegan-train

# Run specific workload with custom parameters
docker-compose run stylegan-train train --dataset-path /workspace/datasets/custom --image-size 512 --outdir /workspace/results

# Run TensorBoard for monitoring
docker-compose up stylegan-tensorboard
```

### Docker Compose Services

The `docker-compose.yml` defines several ready-to-use services:

- `stylegan-train`: For standard training
- `stylegan-inference`: For generating images
- `stylegan-finetune`: For fine-tuning pre-trained models
- `stylegan-optimize-latent`: For latent vector optimization
- `stylegan-mixed-precision`: For mixed precision training
- `stylegan-tensorboard`: For visualizing training progress

## Directory Structure

```
enhanced_modules/
├── Dockerfile          # Docker image definition
├── README.md           # This documentation
├── docker-compose.yml  # Docker Compose configuration
├── stylegan_cli.py     # Unified command-line interface
└── tests/              # Unit tests for workload modules
    └── test_workloads.py
```

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- CUDA 11.0+
- Docker (for containerized execution)
- TensorBoard (optional, for experiment tracking)
- Weights & Biases (optional, for experiment tracking)

## Additional Resources

- Original StyleGAN2-ADA repository: [https://github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- Base workload modules: See the `workload_modules` directory