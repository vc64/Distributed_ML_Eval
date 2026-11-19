import torch
import torch.nn as nn
import torch.optim as optim

import load_data
from models import LogisticRegression, MLP

# model = LogisticRegression(54, 7)
model = MLP(54, 7)

train_dataloader, test_dataloader = load_data.load_data()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


running_loss = 0
last_loss = 0
# Warning: Currently it does not have an end condition
for i, data in enumerate(train_dataloader):
    inputs, labels = data
    print(inputs.dtype)
    print(labels.dtype)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels.view(-1).long()) # Needs to be shape (N,) instead of (N,1) and labels has to be a long (or any int?)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 1000 == 999:
        last_loss = running_loss / 1000 # loss per batch
        print(f'  batch {i + 1} loss: {last_loss}')
        running_loss = 0