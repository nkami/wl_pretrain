import sys
import networkx
from networkx import gnm_random_graph, erdos_renyi_graph, weisfeiler_lehman_graph_hash, is_isomorphic
from networkx import Graph
import torch
from torch_geometric.data import Data
import numpy as np
import hashlib
from sortedcontainers import SortedList
import multiprocessing
from multiprocessing import Process, SimpleQueue


PARALLEL_THRESHOLD = 500_000
NUM_CPUS = 4


class NodeHashGroup(object):
    def __init__(self, node_hashes: set, graphs_with_hashes: list):
        self.nodes_hash_group = node_hashes
        self.graphs_with_hashes = graphs_with_hashes
        self.count = len(graphs_with_hashes)

    def __lt__(self, other):
        return self.count < other.count


def networkx_to_torch(graph: Graph, labels, feature_scaling=1.):
    nodes_features = torch.ones((graph.number_of_nodes(), 1)).float() * feature_scaling
    sources, neighbors = [], []
    for node in range(graph.number_of_nodes()):
        for neighbor in graph.adj[node].keys():
            sources.append(node)
            neighbors.append(neighbor)
    edge_index = torch.from_numpy(np.array([sources, neighbors])).long()
    torch_labels = torch.from_numpy(labels).long()
    return Data(x=nodes_features, edge_index=edge_index, y=torch_labels)


def torch_to_networkx(data):
    graph = networkx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        # if to_undirected_upper and u > v:
        #     continue
        graph.add_edge(u, v)
    return graph


def _wl_converged(nodes_hashes1, nodes_hashes2):
    nodes_partition1, nodes_partition2 = {}, {}
    for cur_key, cur_val in nodes_hashes1.items():
        if cur_val not in nodes_partition1:
            nodes_partition1[cur_val] = [cur_key]
        else:
            nodes_partition1[cur_val].append(cur_key)
    nodes_partition1 = [tuple(cur_set) for cur_set in nodes_partition1.values()]
    nodes_partition1.sort()
    for cur_key, cur_val in nodes_hashes2.items():
        if cur_val not in nodes_partition2:
            nodes_partition2[cur_val] = [cur_key]
        else:
            nodes_partition2[cur_val].append(cur_key)
    nodes_partition2 = [tuple(cur_set) for cur_set in nodes_partition2.values()]
    nodes_partition2.sort()
    return nodes_partition1 == nodes_partition2


def wl_node_hashing(graph: Graph, init_hashes=None, max_iters=None):
    if init_hashes is None:
        prev_nodes_hashes = {node: '0' for node in range(graph.number_of_nodes())}
    else:
        prev_nodes_hashes = init_hashes
    num_iterations = max_iters if max_iters is not None else graph.number_of_nodes()
    for _ in range(num_iterations):
        cur_nodes_hashes = {}
        for cur_node in range(graph.number_of_nodes()):
            cur_neighbors_hashes = [prev_nodes_hashes[cur_node]]
            for neighbor in graph.adj[cur_node].keys():
                cur_neighbors_hashes.append(prev_nodes_hashes[neighbor])
            cur_neighbors_hashes.sort()
            combined_string = ''.join(cur_neighbors_hashes).encode('utf-8')
            hasher = hashlib.sha224()
            hasher.update(combined_string)
            cur_nodes_hashes[cur_node] = hasher.hexdigest()

        prev_nodes_hashes = cur_nodes_hashes
    return prev_nodes_hashes


def _create_hist(hashes: dict, equivalent_hashes: set):
    hist = {}
    for cur_hash in hashes.values():
        if cur_hash in equivalent_hashes:
            hist['0'] = hist.get('0', 0) + 1
        else:
            hist[cur_hash] = hist.get(cur_hash, 0) + 1
    return hist


def _hash_groups_mergeable(first_group: NodeHashGroup, first_group_graphs: list, second_group: NodeHashGroup,
                           second_group_graphs: list, queue: SimpleQueue = None):
    # check that merging hashes does not change WL test outcome
    for first_graph_node_hashes in first_group_graphs:
        cur_first_hist = _create_hist(first_graph_node_hashes, first_group.nodes_hash_group)
        for second_graph_node_hashes in second_group_graphs:
            cur_second_hist = _create_hist(second_graph_node_hashes, second_group.nodes_hash_group)
            hists_are_equal = cur_first_hist == cur_second_hist
            if hists_are_equal:  # merging will make the pair of graphs indistinguishable
                if queue is not None:
                    queue.put(False)
                return False
    if queue is not None:
        queue.put(True)
    return True


def _parallel_hash_groups_mergeable(first_group, second_group, all_node_hashes, chunk_sizes):
    procs = []
    queue = SimpleQueue()
    cur_start, cur_end = 0, chunk_sizes[0]
    second_group_graphs = [all_node_hashes[i] for i in second_group.graphs_with_hashes]
    for cur_cpu in range(len(chunk_sizes)):
        first_group_graphs = [all_node_hashes[i] for i in first_group.graphs_with_hashes[cur_start:cur_end]]
        new_proc = Process(target=_hash_groups_mergeable, args=(first_group, first_group_graphs, second_group,
                                                                second_group_graphs, queue))
        procs.append(new_proc)
        new_proc.start()
        cur_start += chunk_sizes[cur_cpu]
        cur_end += chunk_sizes[min(cur_cpu + 1, len(chunk_sizes) - 1)]
    for cur_proc in procs:
        cur_proc.join()
    results = [queue.get() for _ in range(len(chunk_sizes))]
    # mergeable = len([res for res in results if res == False]) == 0
    # mergeable = all(results)
    return all(results)


def create_vocabulary(graphs, max_iters, target_size, init_hashes):
    all_node_hashes = [wl_node_hashing(graph, max_iters=max_iters, init_hashes=cur_init) for graph, cur_init in zip(graphs, init_hashes)]
    init_node_group_hashes = {}
    for graph_idx, cur_nodes_hash in enumerate(all_node_hashes):
        for cur_hash in cur_nodes_hash.values():
            if cur_hash in init_node_group_hashes:
                init_node_group_hashes[cur_hash].add(graph_idx)
            else:
                init_node_group_hashes[cur_hash] = {graph_idx}
    init_node_group_hashes = [NodeHashGroup({cur_hash}, list(cur_graphs)) for cur_hash, cur_graphs in
                              init_node_group_hashes.items()]
    vocab = SortedList(init_node_group_hashes)
    print(f'the initial number of labels is {len(vocab)}')
    sys.stdout.flush()
    non_mergeable = []
    # last_sizes = None
    while len(vocab) + len(non_mergeable) > target_size and len(vocab) > 1:
        # print((len(vocab), len(non_mergeable), last_sizes))
        sys.stdout.flush()
        merged_group = None
        for idx in range(len(vocab)):
            first_group = vocab[0]
            second_group = vocab[idx]
            # check that node hashes are from different graphs (merging node hashes from the same graph is bad)
            if not set(first_group.graphs_with_hashes).isdisjoint(set(second_group.graphs_with_hashes)):
                continue
            # last_sizes = (len(first_group.graphs_with_hashes), len(second_group.graphs_with_hashes))
            if len(first_group.graphs_with_hashes) * len(second_group.graphs_with_hashes) < PARALLEL_THRESHOLD:
                first_group_graphs = [all_node_hashes[i] for i in first_group.graphs_with_hashes]
                second_group_graphs = [all_node_hashes[i] for i in second_group.graphs_with_hashes]
                mergeable = _hash_groups_mergeable(first_group, first_group_graphs, second_group, second_group_graphs)
            else:
                # num_cpus = multiprocessing.cpu_count()
                num_cpus = NUM_CPUS
                chunk_sizes = [len(first_group.graphs_with_hashes) // num_cpus for _ in range(num_cpus)]
                chunk_sizes[-1] += len(first_group.graphs_with_hashes) - sum(chunk_sizes)
                mergeable = _parallel_hash_groups_mergeable(first_group, second_group, all_node_hashes, chunk_sizes)

            if mergeable:
                all_graphs = first_group.graphs_with_hashes + second_group.graphs_with_hashes
                merged_group = NodeHashGroup(first_group.nodes_hash_group.union(second_group.nodes_hash_group),
                                             all_graphs)
                # update all_node_hashes
                for graph_idx in all_graphs:
                    for cur_node, cur_hash in all_node_hashes[graph_idx].items():
                        if cur_hash in first_group.nodes_hash_group or cur_hash in second_group.nodes_hash_group:
                            hash_in_first_group = next(iter(first_group.nodes_hash_group))
                            all_node_hashes[graph_idx][cur_node] = hash_in_first_group
                del vocab[idx]
                del vocab[0]
                break

        if merged_group is None:
            non_mergeable.append(vocab.pop(0))
        else:
            vocab.add(merged_group)
    vocab.update(non_mergeable)
    final_hash_map = {}
    for cur_label, cur_hash_group in enumerate(vocab):
        for cur_hash in cur_hash_group.nodes_hash_group:
            final_hash_map[cur_hash] = cur_label
    return final_hash_map, all_node_hashes, vocab


if __name__ == '__main__':
    # generate_graphs('random', num_graphs=2, num_nodes=10, num_edges=10)
    # generate_graphs('erdos_renyi', num_graphs=2, num_nodes=10, p=0.5)
    # num_iters = 3
    from networkx import is_connected

    # graphs = [gnm_random_graph(n=25, m=50, seed=_) for _ in range(25_000)]
    # graphs = [g for g in graphs if is_connected(g)]
    # graphs2 = [gnm_random_graph(n=25, m=33, seed=250000 + _) for _ in range(250000)]
    # graphs2 = [g for g in graphs2 if is_connected(g)]
    # graphs = graphs + graphs2
    # graphs = [erdos_renyi_graph(n=7, p=0.5, seed=_) for _ in range(25_000)]
    # for g in graphs:
    #     networkx.set_node_attributes(g, '0', 'features')
    # node_labels = {}
    # graph_hahses = {}
    # for g in graphs:
    #     res = wl_node_hashing(g, max_iters=num_iters)
    #     hash_elements = []
    #     for val in res.values():
    #         node_labels[val] = node_labels.get(val, 0) + 1
    #         hash_elements.append(val)
    #     hash_elements.sort()
    #     combined_string = ''.join(hash_elements).encode('utf-8')
    #     hasher = hashlib.sha224()
    #     hasher.update(combined_string)
    #     cur_hash = hasher.hexdigest()
    #     graph_hahses[cur_hash] = graph_hahses.get(cur_hash, 0) + 1
    # print(len(node_labels), sum([count for count in node_labels.values()]), len(graph_hahses), len(graphs))

    # matches = 0
    # k = 0
    # for i in range(len(graphs)):
    #     for j in range(i, len(graphs)):
    #         if i == j:
    #             continue
    #         k += 1
    #         if k % 1000 == 0:
    #             print(k)
    #         # res1 = wl_node_hashing(graphs[i], max_iters=num_iters)
    #         # res2 = wl_node_hashing(graphs[j], max_iters=num_iters)
    #         hash1 = weisfeiler_lehman_graph_hash(graphs[i], iterations=num_iters, node_attr='features')
    #         hash2 = weisfeiler_lehman_graph_hash(graphs[j], iterations=num_iters, node_attr='features')
    #         if hash1 == hash2:
    #             matches += 1
    #         # networkx_equality = hash1 == hash2
    #         # my_equality = wl_test(res1, res2)
    #         # assert networkx_equality == my_equality
    #         # if my_equality:
    #         #     matches += 1
    # print(matches, k)

    import datetime
    print(NUM_CPUS, PARALLEL_THRESHOLD)
    print(datetime.datetime.now())
    sys.stdout.flush()

    graphs = [erdos_renyi_graph(n=10, p=0.5, seed=_) for _ in range(20_000)]
    final_map, all_nodes_hashes, vocab = create_vocabulary(graphs, max_iters=3, target_size=100)
    print(datetime.datetime.now())
    sys.stdout.flush()
