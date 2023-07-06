import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class Net(torch.nn.Module):
    def __init__(self, leaky_relu_alpha=0.1):
        super(Net, self).__init__()
        self.leaky_relu_alpha = leaky_relu_alpha
        self.conv1 = TransformerConv(130, 50, 4)
        self.conv2 = TransformerConv(200, 25, 4)
        self.conv3 = TransformerConv(100, 10, 4)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)