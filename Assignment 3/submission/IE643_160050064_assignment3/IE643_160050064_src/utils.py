## AUTHOR: Vamsi Krishna Reddy Satti

##################################################################################
# Data loader
##################################################################################


import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset if isinstance(dataset, tuple) else (dataset, )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = self.dataset[0].shape[0]
        self.batches_outstanding = np.ceil(self.dataset_size / self.batch_size).astype(np.long).item()
        self.shuffle_data()

    def __iter__(self):
        return self

    def __len__(self):
        return self.batches_outstanding

    def __next__(self):
        if self.batches_outstanding == 0:
            self.batches_outstanding = np.ceil(self.dataset_size / self.batch_size).astype(np.long).item()  # This helps for next epoch to reuse the same dataloader object
            self.shuffle_data()
            raise StopIteration
        self.batches_outstanding -= 1
        batch = tuple(data[self.batches_outstanding * self.batch_size: (self.batches_outstanding + 1) * self.batch_size] for data in self.dataset)
        return batch if len(batch) > 1 else batch[0]

    def shuffle_data(self):
        if self.shuffle:
            indices = np.random.permutation(self.dataset_size)
            self.dataset = [data[indices] for data in self.dataset]
