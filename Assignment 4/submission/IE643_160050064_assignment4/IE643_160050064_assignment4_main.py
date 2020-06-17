## AUTHOR: Vamsi Krishna Reddy Satti

# ============================================================================
# Filename: main.py
# Example usage: python main.py --data ../natural_images [--checkpoint .pth]
# ============================================================================


import argparse
from IE643_160050064_assignment4_lib.data import NaturalImagesDataset
from IE643_160050064_assignment4_lib.model import Discriminator, Generator
import matplotlib.pyplot as plt
import os
import pprint
import time
import torch
from torch import nn, optim
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--category', type=str, default='person')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--lr-gen', type=float, default=0.0002)
    parser.add_argument('--lr-dis', type=float, default=0.00015)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--val-split', type=float, default=0.0)

    return parser.parse_args()


def main():
    CHANNELS, IMAGE_SIZE = 3, 64
    OUT_DIR = os.path.join('output', time.strftime('%m_%d_%H_%M_%S'))
    OUT_TRANSFORM = transforms.Compose([
                        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
                        transforms.ToPILImage(),
                        transforms.Resize(100),
                    ])

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint) if args.checkpoint else None

    dataset = NaturalImagesDataset(args.data, args.category, \
        transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    indices = torch.randperm(len(dataset))
    split = int(args.val_split * indices.numel())
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, num_workers=4, pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=val_sampler,
        batch_size=4 * args.batch_size, num_workers=4, pin_memory=True
    )

    gen = Generator(CHANNELS, IMAGE_SIZE, args.latent_dim)
    dis = Discriminator(CHANNELS, IMAGE_SIZE)
    start_epoch = 1
    if checkpoint:
        gen.load_state_dict(checkpoint['generator_state_dict'])
        dis.load_state_dict(checkpoint['discriminator_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    gen.to(device)
    dis.to(device)
    
    opt_gen = optim.Adam(gen.parameters(), lr=args.lr_gen, betas=(args.beta1, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=args.lr_dis, betas=(args.beta1, 0.999))
    if checkpoint:
        opt_gen.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        opt_dis.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    loss_fn_val = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    sched_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=10, gamma=1)
    sched_dis = optim.lr_scheduler.StepLR(opt_dis, step_size=10, gamma=1)
    
    losses_gen, losses_dis = [], []
    if checkpoint:
        losses_gen, losses_dis = checkpoint['loss']
    Z = torch.randn(1, args.latent_dim).to(device)

    os.makedirs(OUT_DIR)
    file = open(os.path.join(OUT_DIR, 'log.txt'), 'a')
    pprint.pprint(vars(args), file)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        gen.train()
        dis.train()

        for batch_idx, imgs in enumerate(data_loader):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            real = torch.ones(batch_size, device=device)
            fake = torch.zeros(batch_size, device=device)

            opt_dis.zero_grad()
            loss_dis_real = loss_fn(dis(imgs), real)
            z = torch.randn(batch_size, args.latent_dim).to(device)
            out_gen = gen(z).detach()
            loss_dis_fake = loss_fn(dis(out_gen), fake)
            loss_dis = loss_dis_real + loss_dis_fake
            loss_dis.backward()
            opt_dis.step()

            losses_dis.append(loss_dis.item())
            
            for _ in range(2):
                opt_gen.zero_grad()
                z = torch.randn(batch_size, args.latent_dim).to(device)
                out_gen = gen(z)
                loss_gen = loss_fn(dis(out_gen), real)
                loss_gen.backward()
                opt_gen.step()

            losses_gen.append(loss_gen.item())
            
            if batch_idx % args.log_interval == 0:

                dis.eval()

                log = f"[{epoch}/{start_epoch + args.epochs - 1}][{batch_idx}/{len(data_loader)}] \
                        \t loss_gen: {loss_gen.item():.6f} \
                        \t loss_dis: {loss_dis.item():.6f}"

                if len(val_data_loader) > 0:
                    loss_dis_val = 0
                    with torch.no_grad():
                        for imgs in val_data_loader:
                            imgs = imgs.to(device)
                            batch_size = imgs.size(0)
                            real = torch.ones(batch_size, device=device)
                            loss_dis_val += loss_fn_val(dis(imgs), real).item()
                    loss_dis_val /= len(val_data_loader)
                    log += f"\t loss_dis_val: {loss_dis_val:.6f}"

                print(log)
                print(log, file=file)
        
        gen.eval()
        out_gen = gen(Z).cpu()[0]
        t = OUT_TRANSFORM(out_gen)
        t.save(os.path.join(OUT_DIR, f"{epoch}.png"))
        
        plt.clf()
        plt.plot(losses_gen, label="Generator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(OUT_DIR, f'plot_gen_{epoch}.png'))
        
        plt.clf()
        plt.plot(losses_dis, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(OUT_DIR, f'plot_dis_{epoch}.png'))

        sched_gen.step()
        sched_dis.step()

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': gen.state_dict(),
                'discriminator_state_dict': dis.state_dict(),
                'optimizer_generator_state_dict': opt_gen.state_dict(),
                'optimizer_discriminator_state_dict': opt_dis.state_dict(),
                'loss': (losses_gen, losses_dis),
            }, os.path.join(OUT_DIR, f'checkpoint_{epoch}.pt'))

    file.close()


if __name__ == '__main__':
    main()
