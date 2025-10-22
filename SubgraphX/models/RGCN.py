import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


# RGCN for graph classification
class RGCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args, num_relations):
        super(RGCNNet, self).__init__()
        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden
        self.emb_normlize = model_args.emb_normlize
        self.device = model_args.device
        self.num_gnn_layers = len(self.latent_dim)
        self.num_mlp_layers = len(self.mlp_hidden) + 1
        self.dense_dim = self.latent_dim[-1]
        self.readout_layers = get_readout_layers(model_args.readout)
        self.num_relations = num_relations

        # RGCN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(RGCNConv(input_dim, self.latent_dim[0], 
                                      num_relations=num_relations,
                                      num_bases=model_args.num_bases,
                                      num_blocks=model_args.num_blocks))
        
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(RGCNConv(self.latent_dim[i - 1], self.latent_dim[i], 
                                         num_relations=num_relations,
                                         num_bases=model_args.num_bases,
                                         num_blocks=model_args.num_blocks))
        
        self.gnn_non_linear = nn.ReLU()

        # MLP layers
        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(self.dense_dim * len(self.readout_layers),
                                       model_args.mlp_hidden[0]))
            for i in range(1, self.num_mlp_layers-1):
                self.mlps.append(nn.Linear(self.mlp_hidden[i-1], self.mlp_hidden[i]))
            self.mlps.append(nn.Linear(self.mlp_hidden[-1], output_dim))
        else:
            self.mlps.append(nn.Linear(self.dense_dim * len(self.readout_layers),
                                       output_dim))
        
        self.dropout = nn.Dropout(model_args.dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        
        # RGCN layers
        for i in range(self.num_gnn_layers):
            x = self.gnn_layers[i](x, edge_index, edge_type)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)
        
        emb = x
        
        # Readout layers
        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(emb, batch))
        x = torch.cat(pooled, dim=-1)

        # MLP layers
        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.mlp_non_linear(x)
            x = self.dropout(x)

        logits = self.mlps[-1](x)
        probs = self.Softmax(logits)
        return logits, probs, emb


# RGCN for node classification
class RGCNNet_NC(nn.Module):
    def __init__(self, input_dim, output_dim, model_args, num_relations):
        super(RGCNNet_NC, self).__init__()
        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden
        self.emb_normlize = model_args.emb_normlize
        self.device = model_args.device
        self.concate = model_args.concate
        self.num_gnn_layers = len(self.latent_dim)
        self.num_mlp_layers = len(self.mlp_hidden) + 1
        self.dense_dim = self.latent_dim[-1]
        self.num_relations = num_relations

        # RGCN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(RGCNConv(input_dim, self.latent_dim[0], 
                                      num_relations=num_relations,
                                      num_bases=model_args.num_bases,
                                      num_blocks=model_args.num_blocks))
        
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(RGCNConv(self.latent_dim[i - 1], self.latent_dim[i], 
                                         num_relations=num_relations,
                                         num_bases=model_args.num_bases,
                                         num_blocks=model_args.num_blocks))
        
        self.gnn_non_linear = nn.ReLU()

        # MLP layers
        self.mlps = nn.ModuleList()
        if self.concate:
            mlp_input_dim = self.dense_dim * len(self.latent_dim)
        else:
            mlp_input_dim = self.dense_dim
            
        if self.num_mlp_layers > 1:
            self.mlps.append(nn.Linear(mlp_input_dim, model_args.mlp_hidden[0]))
            for i in range(1, self.num_mlp_layers-1):
                self.mlps.append(nn.Linear(self.mlp_hidden[i-1], self.mlp_hidden[i]))
            self.mlps.append(nn.Linear(self.mlp_hidden[-1], output_dim))
        else:
            self.mlps.append(nn.Linear(mlp_input_dim, output_dim))
        
        self.dropout = nn.Dropout(model_args.dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ELU()

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x_all = []
        
        # RGCN layers
        for i in range(self.num_gnn_layers):
            x = self.gnn_layers[i](x, edge_index, edge_type)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)
            x_all.append(x)
        
        # Get embedding
        if self.concate:
            emb = torch.cat(x_all, dim=-1)
        else:
            emb = x
        
        # MLP layers for classification
        x = emb
        for i in range(self.num_mlp_layers - 1):
            x = self.mlps[i](x)
            x = self.mlp_non_linear(x)
            x = self.dropout(x)

        logits = self.mlps[-1](x)
        probs = self.Softmax(logits)
        return logits, probs, emb


if __name__ == "__main__":
    from Configures import model_args
    
    # Test RGCN for graph classification
    model_args.num_bases = 4  # Number of basis functions for weight sharing
    model_args.num_blocks = 8  # Number of blocks for block diagonal decomposition
    
    num_relations = 3  # Number of relation types in your graph
    model = RGCNNet(7, 2, model_args, num_relations)
    print("RGCN model created successfully!")
    
    # Test RGCN for node classification
    model_nc = RGCNNet_NC(7, 2, model_args, num_relations)
    print("RGCN_NC model created successfully!")