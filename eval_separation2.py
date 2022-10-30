import matplotlib.pyplot as plt
import hashlib
import numpy as np
import torch
import sys
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from datasets import WLPretrainDataset, NetWorkxGraphsDataset
from models import PretrainGIN
import networkx
from networkx import gnm_random_graph, erdos_renyi_graph, weisfeiler_lehman_graph_hash
import multiprocessing
from multiprocessing import Process, SimpleQueue


# change support for gpu
def eval_separation(model, dataset, queue=None):
    cpu_device = torch.device('cpu')
    model.to(cpu_device)
    model.eval()
    with torch.no_grad():
        graphs_hashes_partition, graphs_hashes_idx_map = {}, {}
        wl_hashes_partition, wl_hashes_idx_map = {}, {}
        cur_idx = 0
        for data, graph_hash in dataset:
            data = data.to(cpu_device)
            feats, res = model(data)
            res = torch.argmax(res, dim=1).tolist()
            node_hashes = sorted([e for e in res])
            node_hashes = [str(num) for num in node_hashes]
            combined_string = ''.join(node_hashes).encode('utf-8')
            hasher = hashlib.sha224()
            hasher.update(combined_string)
            cur_hash = hasher.hexdigest()
            graphs_hashes_idx_map[cur_idx] = cur_hash
            if cur_hash in graphs_hashes_partition:
                graphs_hashes_partition[cur_hash].add(cur_idx)
            else:
                graphs_hashes_partition[cur_hash] = {cur_idx}

            wl_hashes_idx_map[cur_idx] = graph_hash
            if graph_hash in wl_hashes_partition:
                wl_hashes_partition[graph_hash].add(cur_idx)
            else:
                wl_hashes_partition[graph_hash] = {cur_idx}
            cur_idx += 1

        numerical_errors, num_mistakes = 0, 0
        diff_memory = set()
        for cur_idx in range(len(dataset)):
            if cur_idx in diff_memory:
                continue
            gin_group = graphs_hashes_partition[graphs_hashes_idx_map[cur_idx]]
            wl_group = wl_hashes_partition[wl_hashes_idx_map[cur_idx]]
            # gin_group should be >= wl_group if no numerical errors
            gin_minus_wl = gin_group.difference(wl_group)
            gin_wl_intersection = gin_group.intersection(wl_group)
            num_mistakes += len(gin_minus_wl) * len(gin_wl_intersection)
            # numerical errors occur because we know that GIN cant differentiate between graphs that are
            # the same according to WL test
            wl_minus_gin = wl_group.difference(gin_group)
            numerical_errors += len(wl_minus_gin)

            # update memory
            diff_memory.update(gin_wl_intersection)
            diff_memory.update(wl_minus_gin)

        if queue is not None:
            queue.put((num_mistakes, numerical_errors))
        # return graphs_hashes, wl_hashes, num_mistakes, numerical_errors
        return num_mistakes, numerical_errors


def calc_graph_conflicts(hidden_channels_sizes, num_labels, num_graphs, num_seeds, act_func, num_nodes, p_vals,
                         graph_axs, axs_idx, x_label, y_label, integer_only=False):
    graph_axs[axs_idx].set_xlabel(x_label)
    graph_axs[axs_idx].set_ylabel(y_label)
    iterate_nodes = True if isinstance(num_nodes, list) else False
    for cur_channel_size in hidden_channels_sizes:
        error_rate_avg = []
        graph_val_to_iterate = num_nodes if iterate_nodes else p_vals
        for cur_val in graph_val_to_iterate:
            if iterate_nodes:
                graphs = [erdos_renyi_graph(n=cur_val, p=p_vals, seed=_) for _ in range(num_graphs)]
            else:
                graphs = [erdos_renyi_graph(n=num_nodes, p=cur_val, seed=_) for _ in range(num_graphs)]
            for g in graphs:
                networkx.set_node_attributes(g, '0', 'features')
            feature_scaling = 1 / cur_val if iterate_nodes else 1 / num_nodes
            # feature_scaling = 1.
            ds = NetWorkxGraphsDataset(graphs, feature_scaling)
            procs = []
            cur_results = []
            queue = SimpleQueue()
            for seed in range(num_seeds):
                torch.manual_seed(seed)
                model = PretrainGIN(in_channels=1, out_channels=cur_channel_size, num_layers=3, act_func=act_func,
                                    num_labels=num_labels)
                # model.eval()
                new_proc = Process(target=eval_separation, args=(model, ds, queue))
                procs.append(new_proc)
                new_proc.start()
                # num_mistakes, numerical_errors = eval_separation(model, ds)
                # cur_results.append((num_mistakes, numerical_errors))
            for cur_proc in procs:
                cur_proc.join()
            cur_results = [queue.get() for _ in range(num_seeds)]
            cur_error_rate = [res[0] / (num_graphs ** 2) for res in cur_results]
            print(f'channels: {cur_channel_size}, error rates: {cur_error_rate}, '
                  f'numerical errors: {[res[1] for res in cur_results]}')
            sys.stdout.flush()
            # error_rate_avg.append((sum(cur_num_mistakes) / len(cur_num_mistakes)) / num_graphs)
            error_rate_avg.append(sum(cur_error_rate) / len(cur_error_rate))
        points = graph_axs[axs_idx].scatter(graph_val_to_iterate, error_rate_avg)
        points.set_label(f'GIN with {cur_channel_size} channels')
    graph_axs[axs_idx].legend(loc="upper right")
    if integer_only:
        graph_axs[axs_idx].xaxis.get_major_locator().set_params(integer=True)


def calc_graph_conflicts_pretrain(hidden_channels_sizes, num_labels, num_train_graphs, act_func, num_test_graphs, lr,
                                  bs, num_epochs, num_nodes, p_val, graph_axs, axs_idx):
    graph_axs[axs_idx].set_xlabel('Epoch')
    graph_axs[axs_idx].set_ylabel('1-WL Test Conflicts Rate')
    train_graphs = [erdos_renyi_graph(n=num_nodes, p=p_val, seed=s) for s in range(num_train_graphs)]
    for g in train_graphs:
        networkx.set_node_attributes(g, '0', 'features')
    train_ds = WLPretrainDataset(train_graphs, target_size=num_labels, max_iters=3)
    print(f'num labels: {train_ds.max_label_neurons}')
    test_graphs = [erdos_renyi_graph(n=num_nodes, p=p_val, seed=num_train_graphs + s) for s in range(num_test_graphs)]
    for g in test_graphs:
        networkx.set_node_attributes(g, '0', 'features')
    test_ds = NetWorkxGraphsDataset(test_graphs)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for num_channels in hidden_channels_sizes:
        model = PretrainGIN(in_channels=1, out_channels=num_channels, num_layers=3, act_func=act_func,
                            num_labels=train_ds.max_label_neurons)
        print(f'num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, '
              f'num labels: {num_labels}, num_channels: {num_channels}')

        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.eval()
        mistakes, numerical_errors = eval_separation(model, test_ds)
        conflict_rate = mistakes / (num_test_graphs ** 2)
        print(f'init, conflicts rate: {conflict_rate}')
        sys.stdout.flush()
        # conflict_rate = 0

        conflict_rates, tested_epochs = [conflict_rate], [0]
        for epoch in range(num_epochs):
            model.train()
            model.to(device)
            for batch in train_dl:
                optimizer.zero_grad()
                batch = batch.to(device)
                out_features, out_labels = model(batch)
                loss = loss_fn(out_labels, batch.y.long())
                loss.backward()
                optimizer.step()
            # sys.stdout.flush()

            if (epoch + 1) % 5 == 0:
                model.eval()
                mistakes, numerical_errors = eval_separation(model, test_ds)
                tested_epochs.append(epoch + 1)
                conflict_rate = mistakes / (num_test_graphs ** 2)
                conflict_rates.append(conflict_rate)
                print(f'epoch {epoch}, last_loss {loss}, conflicts rate: {conflict_rate}')
                sys.stdout.flush()
            # if epoch % 100 == 0:
            #     print(epoch)
            #     sys.stdout.flush()
        plot, = graph_axs[axs_idx].plot(tested_epochs, conflict_rates)
        plot.set_label(f'GIN with {num_channels} channels')
    graph_axs[axs_idx].legend(loc="upper right")
    graph_axs[axs_idx].xaxis.get_major_locator().set_params(integer=True)


if __name__ == '__main__':
    num_graphs = 10_000
    act_func = 'tanh'
    num_seeds = 5
    num_labels = 200
    num_epochs = 200
    convergence_nodes, convergence_p = 8, 0.5
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(7.5)
    fig.set_figwidth(25)
    print(f'num graphs: {num_graphs}, num_seeds: {num_seeds}, num_labels: {num_labels}')
    print(f'available cpus: {multiprocessing.cpu_count()}')
    sys.stdout.flush()
    hidden_channels_sizes = [4, 8, 16, 32]
    # hidden_channels_sizes = [64]
    num_nodes = [6, 7, 8, 9, 10]
    p_values = np.linspace(0.1, 0.9, 9)

    calc_graph_conflicts(hidden_channels_sizes, num_labels=num_labels, num_graphs=num_graphs, num_seeds=num_seeds,
                         act_func=act_func, num_nodes=num_nodes, p_vals=0.5, graph_axs=axs, axs_idx=0, x_label='Number of Nodes',
                         y_label='1-WL Test Conflicts Rate', integer_only=True)
    print('finished first plot')
    calc_graph_conflicts(hidden_channels_sizes, num_labels=num_labels, num_graphs=num_graphs, num_seeds=num_seeds,
                         act_func=act_func, num_nodes=convergence_nodes, p_vals=p_values, graph_axs=axs, axs_idx=1,
                         x_label='P', y_label='1-WL Test Conflicts Rate')
    print('finished second plot')

    calc_graph_conflicts_pretrain(hidden_channels_sizes, num_labels=num_labels, num_train_graphs=10_000,
                                  act_func=act_func, num_test_graphs=10_00, lr=0.0001, bs=512, num_epochs=num_epochs,
                                  num_nodes=convergence_nodes, p_val=convergence_p, graph_axs=axs, axs_idx=2)
    fig_name = f'./sep_wl_res_{num_graphs}_{act_func}_s{num_seeds}_e{num_epochs}_t{convergence_nodes}_p{convergence_p}_l{num_labels}.png'
    print(fig_name)
    sys.stdout.flush()
    plt.savefig(fig_name)
