import random
import numpy as np
import torch
from dataset import HW3Dataset
from model import Net
from torch_geometric.utils import degree
from tqdm import tqdm
from matplotlib import pyplot as plt

# set all the random seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALPHA = 0.05
LR = 0.01
WEIGHT_DECAY = 1e-5
EPOCHS = 5000
SCALER = ["Standard", "MinMax"][0]

# YEARS_MEAN = 2014.2371
# YEARS_STD = 3.4610
# DEGREE_MEAN = 4.4429
# DEGREE_STD = 30.4861

if __name__ == '__main__':
    model_name = "TransformerConv_Model"

    # Load dataset
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    # add year and degree to x with normalization
    years = data.node_year.squeeze(1)
    degrees = degree(data.edge_index[1], num_nodes=100000)  # num_nodes is the total number of nodes in the graph

    if SCALER == "MinMax":
        years = (years - years.min()) / (years.max() - years.min()) # min max scale years
        degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min()) # min max scale degrees
    elif SCALER == "Standard":
        years = (years - torch.mean(years.float())) / torch.std(years.float())
        degrees = (degrees - degrees.mean()) / degrees.std()

    data.x = torch.cat([data.x, years.reshape(-1, 1)], dim=1)
    data.x = torch.cat([data.x, degrees.reshape(-1, 1)], dim=1)
    data.to(device)

    model = Net(ALPHA).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        train_loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())
        test_loss = criterion(out[data.val_mask], data.y[data.val_mask].squeeze())
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        _, pred = model(data).max(dim=1)
        train_acc = (pred[data.train_mask].eq(data.y[data.train_mask].squeeze()).sum().item()) / len(data.train_mask)
        test_acc = (pred[data.val_mask].eq(data.y[data.val_mask].squeeze()).sum().item()) / len(data.val_mask)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{model_name}.pth')

    print(best_acc)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses, label='train')
    ax[0].plot(test_losses, label='val')
    ax[0].legend()
    ax[0].set_title("Train & Val Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[1].plot(train_accs, label='train')
    ax[1].plot(test_accs, label='val')
    ax[1].legend()
    ax[1].set_title("Train & Val Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    plt.savefig(f"{model_name}.png")