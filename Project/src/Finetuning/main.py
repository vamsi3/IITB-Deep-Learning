import config as cfg
from data import IMDBDataset
import torch
from torch import nn, optim
from transformers import BertForSequenceClassification


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = IMDBDataset()
    indices = torch.randperm(len(dataset))
    split = int(cfg.VAL_SPLIT * indices.numel())
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    data_loaders = {
        'train': torch.utils.data.DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=2, pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            dataset,
            sampler=val_sampler,
            batch_size=cfg.TEST_BATCH_SIZE, num_workers=2, pin_memory=True
        )
    }
    dataset_sizes = {
        'train': len(train_indices),
        'val': len(val_indices)
    }

    model = BertForSequenceClassification.from_pretrained(cfg.PRETRAINED_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        [
            {
                "params":model.bert.parameters(),
                "lr": cfg.LR_BERT
            },
            {
                "params":model.classifier.parameters(),
                "lr": cfg.LR_CLASSIFIER
            }
        ]
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SCHED_STEP_SIZE, gamma=cfg.SCHED_DECAY)

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + cfg.EPOCHS):
        model.train()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            sentiment_corrects = 0

            for batch_idx, (inputs, sentiment) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                sentiment = sentiment.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)[0]
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

                if batch_idx % 50 == 0 and phase == 'train':
                    print(f"[Epoch {epoch}/{start_epoch + cfg.EPOCHS - 1}] [Batch {batch_idx}/{len(data_loaders[phase])}]\t\tLoss: {running_loss / ((batch_idx + 1) * cfg.TRAIN_BATCH_SIZE)}")

            epoch_loss = running_loss / dataset_sizes[phase]
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print(f'[{phase.upper()}]\ttotal loss: {epoch_loss:.4f}\tsentiment_acc: {sentiment_acc:.4f}')

            if phase == 'train':
                scheduler.step()


if __name__ == '__main__':
    main()
