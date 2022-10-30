import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool

act_funcs = {'relu': nn.ReLU, 'tanh': nn.Tanh}


class PretrainGIN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_labels, act_func='relu', clf_head=None):
        super(PretrainGIN, self).__init__()
        # self.gin = GIN(in_channels=in_channels, hidden_channels=out_channels, num_layers=num_layers,
        #                out_channels=out_channels, act=act_func)
        assert num_layers >= 1
        channel_list = [out_channels] * num_layers
        channel_list = [in_channels] + channel_list
        channels_inps, channels_outs = channel_list[:-1], channel_list[1:]

        self.gin_layers = nn.ModuleList()
        for cur_inp, cur_out in zip(channels_inps, channels_outs):
            self.gin_layers.append(GINConv(nn.Sequential(nn.Linear(cur_inp, cur_out),
                                                         # nn.ReLU(),
                                                         act_funcs[act_func](),
                                                         nn.Linear(cur_out, cur_out))))

        if clf_head is None:
            self.clf_head = nn.Linear(out_channels, num_labels)
        else:
            self.clf_head = clf_head

    def forward(self, data):
        # graph_features = self.gin(data.x, data.edge_index)
        graph_features, edge_index = data.x, data.edge_index
        for cur_layer in self.gin_layers:
            graph_features = cur_layer(graph_features, edge_index)
        # graph_features = self.gin(data.x, data.edge_index)
        return graph_features, self.clf_head(graph_features)


class FineTuneGIN(nn.Module):
    def __init__(self, pretrained_backbone, clf_head):
        super(FineTuneGIN, self).__init__()
        self.pretrained_backbone = pretrained_backbone
        self.clf_head = clf_head

    def forward(self, data):
        node_features, _ = self.pretrained_backbone(data)
        graph_features = global_add_pool(node_features, data.batch)
        return self.clf_head(graph_features)

# if __name__ == '__main__':
#     from torch_geometric.data import Data
#     import torch
#     edge_index = torch.tensor([[0, 1, 1, 2],
#                                [1, 0, 2, 1]], dtype=torch.long)
#     x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#     data = Data(x=x, edge_index=edge_index)
#     a = PretrainGIN(in_channels=1, out_channels=15, num_layers=3, num_labels=100)
#     b = a(data)
#     c = 1