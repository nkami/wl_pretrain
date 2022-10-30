from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch_geometric
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
from models import PretrainGIN, FineTuneGIN

# the code is mostly copied from: https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py

cls_criterion = torch.nn.BCEWithLogitsLoss()


def train(model, device, loader, optimizer):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        batch.x = batch.x.float()

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


# from torch_geometric.graphgym.models.layer import LayerConfig, MLP
from torch_geometric.nn import GIN, global_add_pool, MLP
class TMP(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.gnn = GIN(9, 600, 5, dropout=0.1, jk=None)
        # classifier_config = LayerConfig(num_layers=3, dim_in=600, dim_out=num_tasks, final_act=False, dropout=0.1)
        self.classifier = MLP([600, 256, 256, num_tasks])

    def forward(self, batch, fp_emb: bool = False):
        x, edge_index, batch_ind = batch.x.float(), batch.edge_index, batch.batch
        h = self.gnn(x, batch.edge_index)
        feats = global_add_pool(h, batch_ind)  # N, hidden_channels
        return self.classifier(feats)



def load_model(path, num_tasks):
    if path == '':
        clf_head = nn.Linear(256, num_tasks)
        backbone = PretrainGIN(in_channels=9, out_channels=256, num_layers=3, num_labels=1)
        model = FineTuneGIN(pretrained_backbone=backbone, clf_head=clf_head)
        # model = TMP(num_tasks)
    else:
        clf_head = nn.Sequential(nn.Linear(256, 1122),
                                 nn.ReLU(),
                                 nn.Linear(1122, 1122))
        backbone = PretrainGIN(in_channels=9, out_channels=256, num_layers=3, num_labels=1, clf_head=clf_head)
        backbone.load_state_dict(torch.load(path))
        clf_head2 = nn.Linear(256, num_tasks)
        model = FineTuneGIN(pretrained_backbone=backbone, clf_head=clf_head2)
    return model


if __name__ == '__main__':
    # ogb_dataset_choices = ['ogbg-molpcba', 'ogbg-molmuv', 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp',
    #                        'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molsider', 'ogbg-molclintox', 'pcqm4m-v2']
    ogb_dataset_choices = ['ogbg-molbace', 'ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molsider',
                           'ogbg-molclintox']
    # dataset = PygGraphPropPredDataset(name='ogbg-molbace')
    # ogb_dataset_choices = ['ogbg-molbace', 'ogbg-molbbbp']
    # ogb_dataset_choices = ['ogbg-molhiv']
    # print(ogb_dataset_choices)
    # sys.stdout.flush()
    batch_size = 64
    num_epochs = 20
    num_workers = 1
    # model_path = './pretrained_models/2022_29_10_10_14_10/model_epoch200.pt'
    model_path = ''
    print(f'using model path: {model_path}')
    sys.stdout.flush()
    # 2022_29_10_10_14_10
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    for cur_choice in ogb_dataset_choices:
        print(f'Starting to fine tune on {cur_choice}')
        sys.stdout.flush()
        dataset = PygGraphPropPredDataset(name=cur_choice)
        if dataset.task_type != 'binary classification':
            print(f'supporting only classification tasks, but ds is {dataset.task_type}')
            exit()
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(cur_choice)
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_results, test_results = [], []
        for cur_seed in range(10):
            torch.manual_seed(cur_seed)
            model = load_model(model_path, dataset.num_tasks)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            valid_curve = []
            test_curve = []
            train_curve = []
            for epoch in range(1, num_epochs + 1):
                print("=====Epoch {}".format(epoch))
                # print('Training...')
                sys.stdout.flush()
                train(model, device, train_loader, optimizer)

                # print('Evaluating...')
                sys.stdout.flush()
                train_perf = eval(model, device, train_loader, evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator)
                test_perf = eval(model, device, test_loader, evaluator)

                print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
                sys.stdout.flush()

                train_curve.append(train_perf[dataset.eval_metric])
                valid_curve.append(valid_perf[dataset.eval_metric])
                test_curve.append(test_perf[dataset.eval_metric])

            best_val_epoch = np.argmax(np.array(valid_curve))
            best_train = max(train_curve)

            # print(f'Finished training on {cur_choice}!')
            print(f'Best validation score in seed {cur_seed}: {valid_curve[best_val_epoch]}')
            print(f'Test score in seed {cur_seed}: {test_curve[best_val_epoch]}')
            sys.stdout.flush()

            val_results.append(valid_curve[best_val_epoch])
            test_results.append(test_curve[best_val_epoch])
        print('*' * 0)
        avg_val = sum(val_results) / len(val_results)
        avg_test = sum(test_results) / len(test_results)
        print(f'Finished {cur_choice}!')
        print(f'best vals: {val_results}, best tests: {test_results}, avg val: {avg_val}, avg test: {avg_test}')
        print('*' * 30)
        sys.stdout.flush()