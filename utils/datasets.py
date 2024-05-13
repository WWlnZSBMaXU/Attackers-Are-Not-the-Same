from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

feature_sizes = []

class Criteo(Dataset):
    '''
    To load Criteo dataset.
    '''
    def __init__(self, root="./dataset", train=True, balanced=True, **kwargs):
        self.train = train
        root = os.path.join(root, "criteo")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the Criteo dataset to {} first! Download: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz".format(root))
        
        # sample data
        if balanced:
            file_out = "train_sampled_balanced.txt" if train else "test_sampled_balanced.txt"
        else:
            file_out = "train_sampled.txt" if train else "test_sampled.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train.txt" if train else "test.txt"

            if balanced:
                # balance the dataset
                self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', nrows=1000000, index_col=None)
                data_1, data_0 = [], []
                num_1, num_0 = 0, 0
                for i in range(len(self.csv_data)):
                    if num_1 == 30000 and num_0 == 30000:
                        break
                    if self.csv_data.iloc[i].values[0] == 1:
                        if num_1 == 30000:
                            continue
                        data_1.append(self.csv_data.iloc[i].values)
                        num_1 += 1
                    elif self.csv_data.iloc[i].values[0] == 0:
                        if num_0 == 30000:
                            continue
                        data_0.append(self.csv_data.iloc[i].values)
                        num_0 += 1
                data_all = []
                for d0, d1 in zip(data_0, data_1):
                    data_all.append(d0)
                    data_all.append(d1)
                self.csv_data = pd.DataFrame(data_all)
            else:
                self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', nrows=60000, index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                if idx > 0 and idx <= 13:
                    self.csv_data[col] = self.csv_data[col].fillna(0,)
                elif idx >= 14:
                    self.csv_data[col] = self.csv_data[col].fillna('-1',)

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset sampling completed.")
        
        # process data
        if balanced:
            file_out = "train_processed_balanced.txt" if train else "test_processed_balanced.txt"
        else:
            file_out = "train_processed.txt" if train else "test_processed.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            if balanced:
                file_in = "train_sampled_balanced.txt" if train else "test_sampled_balanced.txt"
            else:
                file_in = "train_sampled.txt" if train else "test_sampled.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                le = LabelEncoder()
                le.fit(self.csv_data[col])  # np.concatenate()
                self.csv_data[col] = le.transform(self.csv_data[col])

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset processing completed.")

        self.csv_data = pd.read_csv(outpath, sep='\t', index_col=None)
        if train:
            global feature_sizes
            feature_sizes.clear()
            cols = self.csv_data.columns.values
            for col in cols:
                feature_sizes.append(len(self.csv_data[col].value_counts()))
            feature_sizes.pop(0)  # do not contain label


    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        x = self.csv_data.iloc[idx].values
        return x[1:], int(x[0])


datasets_choices = [
    "mnist",
    "fashionmnist",
    "cifar10",
    "cifar100",
    "criteo"
]

datasets_name = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "criteo": "Criteo"
}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "criteo": Criteo
}

datasets_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "criteo": 2
}

transforms_default = {
    "mnist": transforms.Compose([transforms.ToTensor()]),
    "fashionmnist": transforms.Compose([transforms.ToTensor()]),
    "cifar10": transforms.Compose([transforms.ToTensor()]),
    "cifar100": transforms.Compose([ transforms.ToTensor()]),
    "criteo": None
}

transforms_augment = {
    "mnist": transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "fashionmnist": transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "criteo": None
}