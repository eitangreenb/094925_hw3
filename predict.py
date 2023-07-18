import torch
import pandas as pd
from dataset import HW3Dataset
from model import Net
from torch_geometric.utils import degree


PATH_TO_MODEL = "TransformerConv_Model.pth"
ALPHA = 0.05
LR = 0.01
WEIGHT_DECAY = 1e-5

YEARS_MEAN = 2014.2371
YEARS_STD = 3.4610
DEGREE_MEAN = 4.4429
DEGREE_STD = 30.4861

if __name__ == '__main__':
    # Load dataset
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    # add years and degrees to x with normalization
    years = data.node_year.squeeze(1)
    degrees = degree(data.edge_index[1], num_nodes=data.x.shape[0])
    years = (years - YEARS_MEAN) / YEARS_STD
    degrees = (degrees - DEGREE_MEAN) / DEGREE_STD
    data.x = torch.cat([data.x, years.reshape(-1, 1)], dim=1)
    data.x = torch.cat([data.x, degrees.reshape(-1, 1)], dim=1)

    # Load model
    model = Net(ALPHA)
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model.eval()
    _, pred = model(data).max(dim=1)
    pred = pred.tolist()
    
    # Save prediction
    df = pd.DataFrame({'idx': list(range(len(pred))), 'prediction': pred})
    df.to_csv('prediction.csv', index=False)
