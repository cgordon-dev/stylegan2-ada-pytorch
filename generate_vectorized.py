#!/usr/bin/env python3
# generate_vectorized.py
#
# This script generates images in a single batched operation.
# It vectorizes the latent vector computation to accelerate generation.

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

def num_range(s: str) -> List[int]:
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(',')]

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (for conditional networks)', default=None)
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Output directory', required=True)
def generate_vectorized(network_pkl: str, seeds: List[int], truncation_psi: float,
                        class_idx: Optional[int], noise_mode: str, outdir: str):
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    os.makedirs(outdir, exist_ok=True)
    
    # Prepare labels for conditional networks
    num_images = len(seeds)
    label = torch.zeros([num_images, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('For conditional networks, please specify --class')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('Warning: --class is ignored for unconditional networks')
    
    # Vectorize latent codes from all seeds
    latent_vectors = []
    for seed in seeds:
        z = np.random.RandomState(seed).randn(1, G.z_dim)
        latent_vectors.append(z)
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    z_batch = torch.from_numpy(latent_vectors).to(device)
    
    # Generate images in one forward pass
    imgs = G(z_batch, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgs_np = imgs.cpu().numpy()
    
    # Save each image individually
    for idx, seed in enumerate(seeds):
        filename = os.path.join(outdir, f'seed{seed:04d}.png')
        PIL.Image.fromarray(imgs_np[idx], 'RGB').save(filename)
        print(f'Saved image for seed {seed} to {filename}')

if __name__ == "__main__":
    generate_vectorized()