#!/usr/bin/env python3
# finetune_cifar10.py
#
# Copyright (c) 2021, NVIDIA CORPORATION.
# All rights reserved.
#
# This script fine-tunes a pretrained StyleGAN2-ADA network on the CIFAR-10 dataset.
# It loads both the generator and discriminator from a pickle file,
# and trains them on CIFAR-10 using a basic GAN training loop.

import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import dnnlib
import legacy

# Simple GAN losses
def discriminator_loss(real_logits, fake_logits):
    # Using non-saturating loss for D.
    real_loss = torch.nn.functional.softplus(-real_logits).mean()
    fake_loss = torch.nn.functional.softplus(fake_logits).mean()
    return real_loss + fake_loss

def generator_loss(fake_logits):
    return torch.nn.functional.softplus(-fake_logits).mean()

@click.command()
@click.option('--network', 'network_pkl', required=True, help='Pretrained network pickle filename')
@click.option('--data-dir', required=True, help='Directory containing CIFAR-10 data (or will be downloaded)')
@click.option('--outdir', required=True, help='Output directory for checkpoints')
@click.option('--batch-size', type=int, default=64, help='Batch size for fine-tuning')
@click.option('--lr', type=float, default=0.002, help='Learning rate')
@click.option('--epochs', type=int, default=10, help='Number of training epochs')
@click.option('--save-interval', type=int, default=500, help='Save checkpoint every N iterations')
def finetune(network_pkl, data_dir, outdir, batch_size, lr, epochs, save_interval):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained networks
    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
    G = network['G_ema'].to(device)
    D = network['D'].to(device)
    
    # Set networks to train mode
    G.train()
    D.train()
    
    # Prepare CIFAR-10 dataset (32x32 images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.99))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))
    
    os.makedirs(outdir, exist_ok=True)
    iteration = 0

    # Training loop
    for epoch in range(epochs):
        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            # Update Discriminator
            optimizer_D.zero_grad()
            # Real images forward
            real_logits = D(real_imgs)
            # Fake images forward
            z = torch.randn(real_imgs.size(0), G.z_dim, device=device)
            # Create dummy labels for conditional networks if needed
            label = torch.zeros(real_imgs.size(0), G.c_dim, device=device)
            fake_imgs = G(z, label)
            fake_logits = D(fake_imgs.detach())
            loss_D = discriminator_loss(real_logits, fake_logits)
            loss_D.backward()
            optimizer_D.step()

            # Update Generator
            optimizer_G.zero_grad()
            fake_logits = D(fake_imgs)
            loss_G = generator_loss(fake_logits)
            loss_G.backward()
            optimizer_G.step()

            if iteration % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Iteration {iteration}: Loss_D = {loss_D.item():.4f}, Loss_G = {loss_G.item():.4f}")
            if iteration % save_interval == 0:
                ckpt_path = os.path.join(outdir, f'ckpt_{iteration:06d}.pt')
                torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'iter': iteration}, ckpt_path)
            iteration += 1

    # Save final checkpoints
    torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'iter': iteration}, os.path.join(outdir, 'final.pt'))
    print("Fine-tuning complete. Checkpoints saved.")

if __name__ == "__main__":
    finetune()