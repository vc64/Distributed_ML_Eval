from sklearn.datasets import fetch_covtype
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

# load dataset using sklearn, first time will take a while due to download time
covtype = fetch_covtype()
print("Data dimensions")
print("Features:", covtype.data.shape)  # (581012, 54)
print("Labels:", covtype.target.shape)  # (581012,)
print()

# loads as pandas dataframes
X, y = fetch_covtype(return_X_y=True, as_frame=True)

# obtained from https://archive.ics.uci.edu/dataset/31/covertype
cover_types = [
    "Spruce/Fir", 
    "Lodgepole Pine", 
    "Ponderosa Pine", 
    "Cottonwood/Willow", 
    "Aspen", 
    "Douglas-fir", 
    "Krummholz"
]

# align labels with indices (0-6)
y = y - 1
print("Distribution of labels:")
print(y.map(lambda i: cover_types[i]).value_counts())
print()

# Cover_Type
# Lodgepole Pine       283301
# Spruce/Fir           211840
# Ponderosa Pine        35754
# Krummholz             20510
# Douglas-fir           17367
# Aspen                  9493
# Cottonwood/Willow      2747

# Note: Lodgepole Pine and Spruce/Fir take up 85% of the dataset
# Honestly a lot of class imbalance if we use all 7 classes

print("Rows with missing values:", X[pd.isna(X).any(axis=1)].shape)
print("Missing labels:", y[pd.isnull(y)].shape)

scaled_vals = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(scaled_vals, y, test_size=0.2)

train_data = X_train.assign(label=y_train)
test_data = X_test.assign(label=y_test)

train_data.to_parquet("data/train_data.parquet")
test_data.to_parquet("data/test_data.parquet")

class CovertypeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_data():
    train_data = pd.read_parquet("data/train_data.parquet")
    test_data = pd.read_parquet("data/test_data.parquet")
    train_dataset = CovertypeDataset(train_data.drop("label", axis=1), train_data["label"])
    test_dataset = CovertypeDataset(test_data.drop("label", axis=1), test_data["label"])
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    return train_dataloader, test_dataloader