from sklearn.datasets import fetch_covtype
import pandas as pd

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

print(scaled_vals.loc[:, (scaled_vals >= 3).any(axis=0)].columns)