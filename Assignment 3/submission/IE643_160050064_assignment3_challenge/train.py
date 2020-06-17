## AUTHOR: Vamsi Krishna Reddy Satti

# ====================================================================================================================
#                                               train.py
#
# Example usage:    python train.py --data data.csv --save-model pretrained.pth
# ====================================================================================================================


import argparse
from utils import get_model, load_dataset
import torch
from torch import nn, optim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=9)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save-model', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = load_dataset(args.data)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    model = get_model()
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('[TRAIN] Epoch: {}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))
    
    torch.save(model.state_dict(), f"{args.save_model}")


if __name__ == '__main__':
    main()
