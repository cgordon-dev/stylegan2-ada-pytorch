#!/usr/bin/env python3
# generate_mixed_precision.py
#
# This script generates images from a pretrained network using mixed precision.
# It is similar to the original generation script, but wraps the generator call in AMP context.

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torch.cuda.amp import autocast

import legacy

def num_range(s: str) -> List[int]:
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    return [int(x) for x in s.split(',')]

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (for conditional networks)', default=None)
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Output directory', required=True)
def generate_mixed_precision(network_pkl: str, seeds: List[int], truncation_psi: float,
                             class_idx: Optional[int], noise_mode: str, outdir: str):
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    os.makedirs(outdir, exist_ok=True)
    
    # Set label for conditional networks
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('For conditional networks, please specify --class')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('Warning: --class is ignored for unconditional networks')
    
    for seed in seeds:
        print(f'Generating image for seed {seed}...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        # Use mixed precision for inference
        with autocast():
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

if __name__ == "__main__":
    generate_mixed_precision()