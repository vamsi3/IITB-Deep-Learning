## AUTHOR: Vamsi Krishna Reddy Satti

# ====================================================================================================================
#                                               test.py
#
# Example usage:    python test.py --data data.csv --load-model pretrained.pth
# ====================================================================================================================


import argparse
from utils import get_model, load_dataset
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--load-model', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = load_dataset(args.data)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    model = get_model()
    model.load_state_dict(torch.load(args.load_model))
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        correct = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('[TEST] Accuracy: {}/{} ({:.2f}%) are correct'.format(
        correct, len(data_loader.dataset),
        100 * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    main()
