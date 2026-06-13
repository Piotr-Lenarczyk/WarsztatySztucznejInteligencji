import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, JumpingKnowledge, global_max_pool, global_mean_pool


class AttentionalGINEModel(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE).
    Incorporates Jumping Knowledge (JK) and global descriptors into its head.
    """
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(cfg.num_layers):
            dim = in_channels if i == 0 else cfg.hidden_dim
            mlp = nn.Sequential(
                nn.Linear(dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=cfg.edge_dim))
            self.bns.append(nn.BatchNorm1d(cfg.hidden_dim))

        # Adaptive Jumping Knowledge Fusion Architecture
        self.jk = JumpingKnowledge(mode='lstm', channels=cfg.hidden_dim, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.Linear((cfg.hidden_dim * 2) + 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        xs = []

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            xs.append(x)

        # Extract hierarchical multi-scale neighborhood matrices
        x_jk = self.jk(xs)

        # Symmetric multi-resolution feature pooling
        pool_mean = global_mean_pool(x_jk, batch)
        pool_max = global_max_pool(x_jk, batch)

        # Inject contextually normalized raw global graph attributes
        x_pooled = torch.cat([pool_mean, pool_max, data.g_desc], dim=-1)

        return self.head(x_pooled).view(-1)