import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

datafolder = '../../../data/corruptmnist/'


class MyAwesomeDataset(Dataset):
    def __init__(self, datafolder, test=False):

        if test:
            test = np.load(os.path.join(datafolder, 'test.npz'))
            images = test['images']
            labels = test['labels']
            self.images = torch.from_numpy(images).float()
            self.labels = torch.from_numpy(labels).long()
            self.data = torch.utils.data.TensorDataset(self.images, self.labels)
        else:
            # load train_0.npz to train_4.npz
            train_0 = np.load(os.path.join(datafolder, 'train_0.npz'))
            train_1 = np.load(os.path.join(datafolder, 'train_1.npz'))
            train_2 = np.load(os.path.join(datafolder, 'train_2.npz'))
            train_3 = np.load(os.path.join(datafolder, 'train_3.npz'))
            train_4 = np.load(os.path.join(datafolder, 'train_4.npz'))
            # make a listt
            train = [train_0, train_1, train_2, train_3, train_4]

            images = [dat['images'] for dat in train]
            labels = [dat['labels'] for dat in train]
            # concatenate the images and labels
            images = np.concatenate(images)
            labels = np.concatenate(labels).reshape(-1, 1)
            self.images = torch.from_numpy(images).float()
            self.labels = torch.from_numpy(labels).long()
            self.data = torch.utils.data.TensorDataset(self.images, self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.expand_dims(self.images[idx], axis=0), self.labels[idx]
        return sample



train = MyAwesomeDataset(datafolder)
test = MyAwesomeDataset(datafolder, test=True)

# We plot the first 10 images in the dataset
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(2, 5, figsize=(10, 4))
# for i in range(10):
#    ax[i//5, i%5].imshow(train[i][0].numpy().squeeze(), cmap='gray')
#    ax[i//5, i%5].set_title(train[i][1].item())
#    ax[i//5, i%5].axis('off')
#    #set the figure title to train
#    fig.suptitle('Train')
# plt.show()

# plot the next 10 images
# fig, ax = plt.subplots(2, 5, figsize=(10, 4))
# for i in range(10):
#    ax[i//5, i%5].imshow(test[i+10][0].numpy().squeeze(), cmap='gray')
#    ax[i//5, i%5].set_title(test[i+10][1].item())
#    ax[i//5, i%5].axis('off')
#    #set the figure title to test
#    fig.suptitle('Test')
# plt.show()

# dataset = torch.load('../../../data/corruptmnist/train_0.npz')
# 'C:/Users/rune7/Documents/GitHub/dtu_mlops/data/corruptmnist/train_0.npz'
