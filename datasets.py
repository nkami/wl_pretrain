import torch
from torch_geometric.data import Dataset
from utils import networkx_to_torch, wl_node_hashing, create_vocabulary
from networkx import gnm_random_graph, erdos_renyi_graph, weisfeiler_lehman_graph_hash
import numpy as np


class NetWorkxGraphsDataset(Dataset):
    def __init__(self, graphs, feature_scaling=1.):
        super().__init__()
        # self.orig_graphs = graphs
        self.graphs = [networkx_to_torch(g, np.zeros((1, 1)), feature_scaling=feature_scaling) for g in graphs]
        self.graph_hashes = [weisfeiler_lehman_graph_hash(graph, iterations=3, node_attr='features') for graph in graphs]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        # return self.graphs[item], self.orig_graphs[item]
        return self.graphs[item], self.graph_hashes[item]


class WLPretrainDataset(Dataset):
    def __init__(self, graphs, target_size, max_iters=3, init_hashes=None, torch_graphs=None):
        super().__init__()
        if init_hashes is None:
            init_hashes = [None for _ in range(len(graphs))]
        final_map, all_nodes_hashes, vocab = create_vocabulary(graphs, max_iters=max_iters, target_size=target_size,
                                                               init_hashes=init_hashes)
        self.node_labels = []
        self.max_label_neurons = len(vocab)
        for cur_hash in all_nodes_hashes:
            cur_node_labels = []
            for cur_node in range(len(cur_hash.keys())):
                cur_node_labels.append(np.array(final_map[cur_hash[cur_node]]))
            self.node_labels.append(np.stack(cur_node_labels))
        if torch_graphs is None:
            self.torch_graphs = [networkx_to_torch(g, l) for g, l in zip(graphs, self.node_labels)]
        else:
            for cur_graph, cur_node_labels in zip(torch_graphs, self.node_labels):
                cur_graph.y = torch.from_numpy(cur_node_labels).long()
                cur_graph.x = cur_graph.x.float()
                # cur_graph.edge_attr = cur_graph.edge_attr.float()
            self.torch_graphs = torch_graphs

    def __len__(self):
        return len(self.torch_graphs)

    def __getitem__(self, item):
        return self.torch_graphs[item]


if __name__ == '__main__':
    num_graphs, n, p = 100, 5, 0.5
    cur_graphs = [erdos_renyi_graph(n=n, p=p, seed=_) for _ in range(num_graphs)]
    ds = WLPretrainDataset(cur_graphs)
    x = 0