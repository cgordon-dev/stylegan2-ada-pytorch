#!/usr/bin/env python3
"""
Unified command-line interface for StyleGAN2-ADA workloads.

This CLI provides a single entry point to all workload modules:
- Training
- Inference
- Fine-tuning optimization
- Latent vector optimization
- Mixed precision optimization

Example usage:
  python stylegan_cli.py train --dataset-path ./datasets/automotive --image-size 256
  python stylegan_cli.py generate --network model.pkl --seeds 0-10 --outdir ./results
  python stylegan_cli.py fine-tune --dataset-path ./datasets/medical --resume ffhq256
  python stylegan_cli.py optimize-latent --network model.pkl --target target.png
  python stylegan_cli.py mixed-precision --dataset-path ./datasets/fashion
"""

import os
import sys
import argparse
import importlib
import logging
from typing import List, Dict, Any, Optional, Callable, Union

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import workload modules
import workload_modules.training_workload as training
import workload_modules.inference_workload as inference
import workload_modules.fine_tuning_optimization as fine_tuning
import workload_modules.latent_vector_optimization as latent_opt
import workload_modules.mixed_precision_optimization as mixed_precision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stylegan_cli.log')
    ]
)
logger = logging.getLogger('stylegan_cli')

# Import and initialize optional tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to the parser."""
    parser.add_argument('--config', type=str, help='Path to configuration file (yaml or json)')
    parser.add_argument('--outdir', type=str, help='Where to save the results')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use')
    parser.add_argument('--dry-run', action='store_true', help='Print options and exit')
    
    # Tracking options
    tracking_group = parser.add_argument_group('tracking')
    tracking_group.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    tracking_group.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    tracking_group.add_argument('--wandb-project', type=str, default='stylegan2-ada', help='W&B project name')
    tracking_group.add_argument('--wandb-entity', type=str, help='W&B entity (team) name')
    tracking_group.add_argument('--log-dir', type=str, default='./logs', help='Directory for logs')
    tracking_group.add_argument('--experiment-name', type=str, help='Name for this experiment run')


def setup_train_args(parser: argparse.ArgumentParser) -> None:
    """Set up training-specific arguments."""
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory or zip file')
    parser.add_argument('--snap', type=int, help='Snapshot interval in training ticks', default=50)
    parser.add_argument('--image-size', type=int, help='Override dataset\'s image resolution')
    parser.add_argument('--batch', type=int, help='Override batch size')
    parser.add_argument('--mirror', type=bool, help='Enable dataset x-flips', default=False)
    parser.add_argument('--kimg', type=int, help='Override training duration in thousands of images')
    parser.add_argument('--resume', type=str, help='Resume from given network pickle')
    parser.add_argument('--use-labels', type=bool, help='Use labels from dataset.json', default=True)
    parser.add_argument('--industry', type=str, help='Industry name for organization')


def setup_generate_args(parser: argparse.ArgumentParser) -> None:
    """Set up inference-specific arguments."""
    parser.add_argument('--network', required=True, help='Network pickle filename')
    parser.add_argument('--seeds', type=str, required=True, help='List of random seeds (e.g., 0-10 or 1,2,3)')
    parser.add_argument('--trunc', type=float, help='Truncation psi', default=1.0)
    parser.add_argument('--class', type=int, dest='class_idx', help='Class label (unconditional if not specified)')
    parser.add_argument('--noise-mode', help='Noise mode', choices=['const', 'random', 'none'], default='const')
    parser.add_argument('--projected-w', help='Projection result file', type=str)
    parser.add_argument('--industry', help='Industry name for organization', type=str)


def setup_fine_tune_args(parser: argparse.ArgumentParser) -> None:
    """Set up fine-tuning-specific arguments."""
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the fine-tuning dataset directory or zip file')
    parser.add_argument('--resume', type=str, required=True, help='Pretrained model to fine-tune (path, URL, or preset name)')
    parser.add_argument('--snap', type=int, help='Snapshot interval in training ticks', default=50)
    parser.add_argument('--image-size', type=int, help='Override dataset\'s image resolution')
    parser.add_argument('--batch', type=int, help='Override batch size')
    parser.add_argument('--mirror', type=bool, help='Enable dataset x-flips', default=False)
    parser.add_argument('--kimg', type=int, help='Override fine-tuning duration in thousands of images', default=1000)
    parser.add_argument('--freezed', type=int, help='Number of layers to freeze in discriminator', default=4)
    parser.add_argument('--lr', type=float, help='Learning rate override')
    parser.add_argument('--use-labels', type=bool, help='Use labels from dataset.json', default=True)
    parser.add_argument('--aug', choices=['noaug', 'ada', 'fixed'], help='Augmentation mode', default='ada')
    parser.add_argument('--p', type=float, help='Augmentation probability for --aug=fixed')
    parser.add_argument('--industry', type=str, help='Industry name for organization')


def setup_latent_opt_args(parser: argparse.ArgumentParser) -> None:
    """Set up latent vector optimization-specific arguments."""
    parser.add_argument('--network', required=True, help='Network pickle filename')
    parser.add_argument('--target', required=True, help='Target image or directory')
    parser.add_argument('--num-steps', type=int, help='Number of optimization steps', default=1000)
    parser.add_argument('--latent-space', choices=['z', 'w', 'w+'], help='Latent space to optimize in', default='w+')
    parser.add_argument('--perceptual-loss', choices=['lpips', 'none'], help='Perceptual loss type', default='lpips')
    parser.add_argument('--initial-lr', type=float, help='Initial learning rate', default=0.1)
    parser.add_argument('--trunc', type=float, dest='truncation_psi', help='Truncation psi', default=0.7)
    parser.add_argument('--noise-mode', choices=['const', 'random', 'none'], help='Noise mode', default='const')
    parser.add_argument('--seed', type=int, help='Random seed', default=0)
    parser.add_argument('--industry', help='Industry name for organization', type=str)


def setup_mixed_precision_args(parser: argparse.ArgumentParser) -> None:
    """Set up mixed precision-specific arguments."""
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset directory or zip file')
    parser.add_argument('--snap', type=int, help='Snapshot interval in training ticks', default=50)
    parser.add_argument('--image-size', type=int, help='Override dataset\'s image resolution')
    parser.add_argument('--batch', type=int, help='Override batch size')
    parser.add_argument('--mirror', type=bool, help='Enable dataset x-flips', default=False)
    parser.add_argument('--kimg', type=int, help='Override training duration in thousands of images')
    parser.add_argument('--resume', type=str, help='Resume from given network pickle')
    parser.add_argument('--nhwc', action='store_true', help='Use NHWC memory format with FP16', default=True)
    parser.add_argument('--mixed-precision-mode', 
                       choices=['default', 'aggressive', 'conservative', 'none'],
                       help='Mixed precision configuration mode', 
                       default='default')
    parser.add_argument('--allow-tf32', action='store_true', help='Allow PyTorch to use TF32 internally', default=True)
    parser.add_argument('--use-labels', type=bool, help='Use labels from dataset.json', default=True)
    parser.add_argument('--industry', type=str, help='Industry name for organization')


def init_tracking(args: argparse.Namespace) -> Dict[str, Any]:
    """Initialize tracking based on arguments."""
    tracking = {}
    
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        # Generate a name based on command and parameters
        if hasattr(args, 'dataset_path'):
            dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
            experiment_name = f"{args.command}_{dataset_name}"
        elif hasattr(args, 'network'):
            network_name = os.path.basename(args.network).split('.')[0]
            experiment_name = f"{args.command}_{network_name}"
        else:
            experiment_name = f"{args.command}_{args.industry or 'default'}"
    
    # Create log directory
    log_dir = os.path.join(args.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard if requested
    if args.tensorboard and TENSORBOARD_AVAILABLE:
        tracking['tensorboard'] = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging enabled. Run 'tensorboard --logdir={log_dir}' to view.")
    
    # Initialize Weights & Biases if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb_config = {k: v for k, v in vars(args).items() if v is not None}
        tracking['wandb'] = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=experiment_name,
            config=wandb_config,
            dir=log_dir
        )
        logger.info(f"Weights & Biases logging enabled. View at {wandb.run.get_url()}")
    
    return tracking


def parse_seeds(seeds_str: str) -> List[int]:
    """Parse seeds string into a list of integers."""
    return inference.num_range(seeds_str)


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='StyleGAN2-ADA Unified CLI')
    subparsers = parser.add_subparsers(dest='command', help='Workload type')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a StyleGAN2-ADA model')
    setup_common_args(train_parser)
    setup_train_args(train_parser)
    
    # Inference command
    generate_parser = subparsers.add_parser('generate', help='Generate images with a trained model')
    setup_common_args(generate_parser)
    setup_generate_args(generate_parser)
    
    # Fine-tuning command
    fine_tune_parser = subparsers.add_parser('fine-tune', help='Fine-tune a pre-trained model')
    setup_common_args(fine_tune_parser)
    setup_fine_tune_args(fine_tune_parser)
    
    # Latent vector optimization command
    latent_opt_parser = subparsers.add_parser('optimize-latent', help='Optimize latent vectors for target images')
    setup_common_args(latent_opt_parser)
    setup_latent_opt_args(latent_opt_parser)
    
    # Mixed precision training command
    mixed_precision_parser = subparsers.add_parser('mixed-precision', help='Train with mixed precision for better performance')
    setup_common_args(mixed_precision_parser)
    setup_mixed_precision_args(mixed_precision_parser)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize tracking if requested
    tracking = {}
    if hasattr(args, 'tensorboard') and (args.tensorboard or args.wandb):
        tracking = init_tracking(args)
    
    # Execute the requested command
    if args.command == 'train':
        logger.info("Starting training workload")
        cmd_args = vars(args)
        cmd_args.pop('command', None)
        cmd_args.pop('tensorboard', None)
        cmd_args.pop('wandb', None)
        cmd_args.pop('wandb_project', None)
        cmd_args.pop('wandb_entity', None)
        cmd_args.pop('log_dir', None)
        cmd_args.pop('experiment_name', None)
        training.main(**cmd_args)
    
    elif args.command == 'generate':
        logger.info("Starting inference workload")
        # Convert seeds string to actual seed list
        if hasattr(args, 'seeds') and args.seeds:
            args.seeds = parse_seeds(args.seeds)
        cmd_args = vars(args)
        cmd_args.pop('command', None)
        cmd_args.pop('tensorboard', None)
        cmd_args.pop('wandb', None)
        cmd_args.pop('wandb_project', None)
        cmd_args.pop('wandb_entity', None)
        cmd_args.pop('log_dir', None)
        cmd_args.pop('experiment_name', None)
        inference.main(**cmd_args)
    
    elif args.command == 'fine-tune':
        logger.info("Starting fine-tuning optimization workload")
        cmd_args = vars(args)
        cmd_args.pop('command', None)
        cmd_args.pop('tensorboard', None)
        cmd_args.pop('wandb', None)
        cmd_args.pop('wandb_project', None)
        cmd_args.pop('wandb_entity', None)
        cmd_args.pop('log_dir', None)
        cmd_args.pop('experiment_name', None)
        fine_tuning.main(**cmd_args)
    
    elif args.command == 'optimize-latent':
        logger.info("Starting latent vector optimization workload")
        cmd_args = vars(args)
        cmd_args.pop('command', None)
        cmd_args.pop('tensorboard', None)
        cmd_args.pop('wandb', None)
        cmd_args.pop('wandb_project', None)
        cmd_args.pop('wandb_entity', None)
        cmd_args.pop('log_dir', None)
        cmd_args.pop('experiment_name', None)
        latent_opt.main(**cmd_args)
    
    elif args.command == 'mixed-precision':
        logger.info("Starting mixed precision optimization workload")
        cmd_args = vars(args)
        cmd_args.pop('command', None)
        cmd_args.pop('tensorboard', None)
        cmd_args.pop('wandb', None)
        cmd_args.pop('wandb_project', None)
        cmd_args.pop('wandb_entity', None)
        cmd_args.pop('log_dir', None)
        cmd_args.pop('experiment_name', None)
        mixed_precision.main(**cmd_args)
        
    # Cleanup tracking resources
    if 'tensorboard' in tracking:
        tracking['tensorboard'].close()
    if 'wandb' in tracking and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()