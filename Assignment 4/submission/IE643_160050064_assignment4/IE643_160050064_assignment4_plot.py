## AUTHOR: Vamsi Krishna Reddy Satti

# ============================================================================
# Filename: plot.py
# Example usage: python plot.py --checkpoint pretrained.pth --output-dir .
# ============================================================================


import argparse
import matplotlib.pyplot as plt
from IE643_160050064_assignment4_lib.model import Generator
import os
import torch
import torchvision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--prefix', type=str, default='IE643_160050064_assignment4')

    return parser.parse_args()


def main():
    CHANNELS, IMAGE_SIZE = 3, 64
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint)

    losses_gen, losses_dis = checkpoint['loss']
    plt.clf()
    plt.title("Generator and Discriminator Loss during training")
    plt.plot(losses_gen, label="Generator")
    plt.plot(losses_dis, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'{args.prefix}_f.png'))

    plt.clf()
    plt.title("Training Objective during training")
    plt.plot(torch.tensor(losses_gen) + torch.tensor(losses_dis))
    plt.xlabel("Training Objective")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.output_dir, f'{args.prefix}_e.png'))
    
    gen = Generator(CHANNELS, IMAGE_SIZE, args.latent_dim)
    gen.load_state_dict(checkpoint['generator_state_dict'])
    gen.to(device).eval()
    Z = torch.randn(50, args.latent_dim).to(device)
    generated_images = gen(Z)
    plt.clf()
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(torchvision.utils.make_grid(generated_images, 
                nrow=10, padding=2, normalize=True).permute(1, 2, 0).cpu())
    plt.savefig(os.path.join(args.output_dir, f'{args.prefix}_g_generated.png'),
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()
