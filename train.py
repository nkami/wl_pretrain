import torch
import torch.nn as nn
from torch.optim import Adam
import torch_geometric
from torch_geometric.data import DataLoader
import networkx
from networkx import erdos_renyi_graph, weisfeiler_lehman_graph_hash, is_isomorphic
import hashlib
from models import PretrainGIN
from datasets import WLPretrainDataset, WLPretrainDataset2, WLPretrainDataset3, NetWorkxGraphsDataset
from utils import smiles_to_networkx
import sys


def eval_separation2(model, dataset, device):
    with torch.no_grad():
        graphs_hashes = {}
        wl_hashes = {}
        i, numerical_errors, num_mistakes = 0, 0, 0
        for data, graph_hash in dataset:
            data = data.to(device)
            feats, res = model(data)
            # res = (res == torch.max(res)).float().tolist()
            res = torch.argmax(res, dim=1).tolist()
            # node_hashes = sorted([bin_to_int(cur_bin) for cur_bin in res])
            node_hashes = sorted([e for e in res])
            node_hashes = [str(num) for num in node_hashes]
            combined_string = ''.join(node_hashes).encode('utf-8')
            hasher = hashlib.sha224()
            hasher.update(combined_string)
            cur_hash = hasher.hexdigest()
            in_graph_hash = False
            if cur_hash in graphs_hashes:
                in_graph_hash = True
                graphs_hashes[cur_hash].append(i)
            else:
                graphs_hashes[cur_hash] = [i]

            in_wl_hash = False
            cur_wl_hash = graph_hash
            # cur_wl_hash = weisfeiler_lehman_graph_hash(graph, iterations=3, node_attr='features')
            if cur_wl_hash in wl_hashes:
                in_wl_hash = True
                wl_hashes[cur_wl_hash].append(i)
            else:
                wl_hashes[cur_wl_hash] = [i]

            if in_graph_hash is False and in_wl_hash is True:
                numerical_errors += 1
            elif in_graph_hash is True and in_wl_hash is False:
                num_mistakes += 1
            i += 1
        return graphs_hashes, wl_hashes, num_mistakes, numerical_errors


if __name__ == '__main__':
    num_train_graphs, n, p = 5000, 8, 0.5
    num_test_graphs = 2000
    num_channels = 32
    train_graphs = [erdos_renyi_graph(n=n, p=p, seed=s) for s in range(num_train_graphs)]
    for g in train_graphs:
        networkx.set_node_attributes(g, '0', 'features')
    train_ds = WLPretrainDataset3(train_graphs, target_size=100, max_iters=3)
    test_graphs = [erdos_renyi_graph(n=n, p=p, seed=num_train_graphs + s) for s in range(num_test_graphs)]
    # test_graphs = [erdos_renyi_graph(n=n, p=p, seed=s) for s in range(num_test_graphs)]
    for g in test_graphs:
        networkx.set_node_attributes(g, '0', 'features')
    test_ds = NetWorkxGraphsDataset(test_graphs)
    bs = 256
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainGIN(in_channels=1, out_channels=num_channels, num_layers=3, num_labels=train_ds.max_label_neurons)
    print(f'num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, '
          f'num labels: {train_ds.max_label_neurons}, num_channels: {num_channels}')
    model.to(device)
    # exit()
    optimizer = Adam(model.parameters(), lr=lr)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    epochs = 10000
    loss = None
    for epoch in range(epochs):
        # model.train()
        # for batch in train_dl:
        #     optimizer.zero_grad()
        #     batch = batch.to(device)
        #     out_features, out_labels = model(batch)
        #     loss = loss_fn(out_labels, batch.y.long())
        #     loss.backward()
        #     # print(loss)
        #     optimizer.step()
        # sys.stdout.flush()

        # if epoch == 10:
        #     k = 0
        #     p = -1
        model.eval()
        _, _, mistakes, numerical_errors = eval_separation2(model, test_ds, device)
        print(f'num mistakes: {mistakes}, numerical_errors: {numerical_errors}, last loss: {loss}')
        sys.stdout.flush()



