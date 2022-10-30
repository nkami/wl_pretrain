from ogb.graphproppred import PygGraphPropPredDataset
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader
from utils import torch_to_networkx
from datasets import WLPretrainDataset
from models import PretrainGIN
import sys
from datetime import datetime
from pathlib import Path
import json


# TODO: take into account edge features
def get_init_hashes(data_list):
    init_hashes = []
    for cur_data in data_list:
        cur_node_hashes = {}
        for cur_node in range(cur_data.num_nodes):
            cur_node_hashes[cur_node] = ''.join([str(cur_feat.item()) for cur_feat in cur_data.x[cur_node, :]])
        init_hashes.append(cur_node_hashes)
    return init_hashes


# use PCBA dataset in moleculenet for pretraining
if __name__ == '__main__':
    hyper_params = {
        'target_num_labels': 1000,
        'num_epochs': 200,
        'bs': 64,
        'num_layers': 3,
        'num_channels': 256,
        'data': 'ogbg-molhiv'
    }
    date_time = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
    model_check_points_dir = f'./pretrained_models/{date_time}'
    model_path = Path(model_check_points_dir)
    model_path.mkdir(exist_ok=True, parents=True)
    print(hyper_params)
    with open(f'{model_check_points_dir}/hyper_params.json', 'w') as f:
        f.write(json.dumps(hyper_params))

    # print(f'batch size: {bs}, num epochs: {num_epochs}, target num labels: {target_num_labels}, '
    #       f'num channels: {num_channels}, num layers: {num_layers}, dataset: {None}')

    # ds = PygGraphPropPredDataset(name='ogbg-moltoxcast')
    # ds = PygGraphPropPredDataset(name='ogbg-molbace')
    ds = PygGraphPropPredDataset(name=hyper_params['data'])
    # ds = PygGraphPropPredDataset(name='ogbg-molpcba')
    networkx_graphs = [torch_to_networkx(cur_data) for cur_data in ds]
    torch_graphs = [cur_data for cur_data in ds]
    init_hashes = get_init_hashes(torch_graphs)
    pretrain_dataset = WLPretrainDataset(networkx_graphs, target_size=hyper_params['target_num_labels'], max_iters=3,
                                         init_hashes=init_hashes, torch_graphs=torch_graphs)
    num_labels = pretrain_dataset.max_label_neurons
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf_head = nn.Sequential(nn.Linear(hyper_params['num_channels'], num_labels),
                             nn.ReLU(),
                             nn.Linear(num_labels, num_labels))
    model = PretrainGIN(in_channels=9, out_channels=hyper_params['num_channels'], num_layers=hyper_params['num_layers'],
                        num_labels=num_labels, clf_head=clf_head)
    print(f'model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} params, num labels: {num_labels}')
    sys.stdout.flush()
    dl = DataLoader(pretrain_dataset, batch_size=hyper_params['bs'], shuffle=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    # save_points = [3, 5, 10, 20]
    for epoch in range(hyper_params['num_epochs']):
        model.train()
        losses = []
        for batch in dl:
            optimizer.zero_grad()
            batch = batch.to(device)
            out_features, out_labels = model(batch)
            loss = loss_fn(out_labels, batch.y.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'epoch {epoch}, average loss: {sum(losses) / len(losses)}')
        sys.stdout.flush()

        # if (epoch + 1) in save_points:
        #     torch.save(model.state_dict(), f'{model_check_points_dir}/model_epoch{epoch + 1}.pt')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'{model_check_points_dir}/model_epoch{epoch + 1}.pt')
