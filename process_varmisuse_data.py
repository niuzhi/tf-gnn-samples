import os
import json
import gzip
import codecs
import math
from tqdm import tqdm
import multiprocessing
import gc
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import numpy as np
from tasks.varmisuse_task import _add_per_subtoken_nodes
from dpu_utils.codeutils import get_language_keywords

colors = {'Child': 'black', 'ReturnsTo': 'forestgreen', 'NextToken': 'red', 'FormalArgName': 'blue',
          'GuardedByNegation': 'gray', 'ComputedFrom': 'chocolate', 'GuardedBy': 'darkorange',
          'LastUse': 'darkolivegreen', 'LastLexicalUse': 'teal', 'LastWrite': 'magenta', 'UsesSubtoken': 'yellow'}

# colors = ["lightcoral", "gray", "lightgray", "firebrick", "red", "chocolate", "darkorange",
#           "moccasin", "gold", "yellow", "darkolivegreen", "chartreuse", "forestgreen", "lime",
#           "mediumaquamarine", "turquoise", "teal", "cadetblue", "dogerblue", "blue", "slateblue",
#           "blueviolet", "magenta", "lightsteelblue"]


def load_jsonl_gz(file_path):
    reader = codecs.getreader('utf-8')
    data = []
    with gzip.open(file_path) as f:
        json_list = list(reader(f))
    data.extend([json.loads(jline) for jline in json_list])
    return data


def save_jsonl_gz(functions, file_name):
    with gzip.GzipFile(file_name, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for entry in functions:
            writer(out_file).write(json.dumps(entry))
            writer(out_file).write('\n')


def get_dataset_num():
    train_samples_len, valid_samples_len, test_samples_len, testonly_samples_len = 0, 0, 0, 0
    files = os.listdir('./data/varmisuse/graphs-train')
    for file in files:
        train_samples_len += len(load_jsonl_gz(os.path.join('./data/varmisuse/graphs-train', file)))
    print('train samples: %d' % train_samples_len)
    files = os.listdir('./data/varmisuse/graphs-valid')
    for file in files:
        valid_samples_len += len(load_jsonl_gz(os.path.join('./data/varmisuse/graphs-valid', file)))
    print('valid samples: %d' % valid_samples_len)
    files = os.listdir('./data/varmisuse/graphs-test')
    for file in files:
        test_samples_len += len(load_jsonl_gz(os.path.join('./data/varmisuse/graphs-test', file)))
    print('test samples: %d' % test_samples_len)
    files = os.listdir('./data/varmisuse/graphs-testonly')
    for file in files:
        testonly_samples_len += len(load_jsonl_gz(os.path.join('./data/varmisuse/graphs-testonly', file)))
    print('testonly samples: %d' % testonly_samples_len)


def add_index():
    files_dict = {}
    files_dict['graphs-train'] = os.listdir('./data/varmisuse/graphs-train')
    files_dict['graphs-valid'] = os.listdir('./data/varmisuse/graphs-valid')
    files_dict['graphs-test'] = os.listdir('./data/varmisuse/graphs-test')
    files_dict['graphs-testonly'] = os.listdir('./data/varmisuse/graphs-testonly')
    index = 0
    for key in files_dict.keys():
        for file in files_dict[key]:
            samples = load_jsonl_gz(os.path.join('./data/varmisuse', key, file))
            for sample in tqdm(samples, desc=key + ':' + file):
                sample['index'] = index
                index += 1
            save_jsonl_gz(samples, os.path.join('./data/varmisuse', key, file))


def build_networkx_graph(index, context_graph, symbol_cands, slot_dummy_node):
    cur_graph = nx.MultiDiGraph()
    node_labels = context_graph['NodeLabels']
    edges = context_graph['Edges']
    node_types = context_graph['NodeTypes']
    symbol_dummy_nodes_correct = dict()
    for symbol_cand in symbol_cands:
        symbol_dummy_node = symbol_cand['SymbolDummyNode']
        symbol_name = symbol_cand['SymbolName']
        is_correct = symbol_cand['IsCorrect']
        symbol_dummy_nodes_correct[symbol_dummy_node] = is_correct
    for key in node_labels.keys():
        value = node_labels[key]
        if key in node_types.keys():
            node_type = node_types[key]
        else:
            node_type = None
        cur_graph.add_node(int(key))
        cur_node = cur_graph.nodes[int(key)]
        cur_node['label'] = bytes(str(key) + ':' + str(value) + ':' + str(node_type), 'utf-8')
        cur_node['type'] = node_type
        cur_node['value'] = bytes(str(value), 'utf-8')
        if int(key) == slot_dummy_node:
            cur_node['fillcolor'] = 'red'
            cur_node['style'] = 'filled'
        if int(key) in symbol_dummy_nodes_correct.keys():
            if symbol_dummy_nodes_correct[int(key)]:
                cur_node['fillcolor'] = 'green'
                cur_node['style'] = 'filled'
                cur_node['IsCorrect'] = True
            else:
                cur_node['fillcolor'] = 'lightsteelblue'
                cur_node['style'] = 'filled'
                cur_node['IsCorrect'] = False
    for edge_type in edges.keys():
        each_type_edges = edges[edge_type]
        for each_type_edge in each_type_edges:
            cur_graph.add_edges_from([(each_type_edge[0], each_type_edge[1], dict(label=edge_type,
                                                                                  color=colors[edge_type]))])
    return cur_graph, symbol_dummy_nodes_correct


def get_next_tokens():
    file_path = '/mnt/data/shangqing1/ptgnn/data/varmisuse/train-out/wox.0.jsonl.gz'
    samples = load_jsonl_gz(file_path)
    for sample in samples:
        file_name = sample['filename']
        index = sample['index']
        context_graph = sample['ContextGraph']
        next_token_edges = context_graph['Edges']['NextToken']
        next_token_nodes = []
        for next_token_edge in next_token_edges:
            next_token_nodes.append(next_token_edge[0])
            next_token_nodes.append(next_token_edge[1])
        next_token_nodes = list(set(next_token_nodes))
        symbol_cands = sample['SymbolCandidates']
        slot_dummy_node = sample['SlotDummyNode']
        cur_graph = build_networkx_graph(index, context_graph, symbol_cands, slot_dummy_node)
        sub_graph = cur_graph.subgraph(next_token_nodes)
        unfrozen_sub_graph = nx.Graph(sub_graph)
        to_remove = [(a, b) for a, b, attrs in unfrozen_sub_graph.edges(data=True) if not (
            attrs['label'] == 'NextToken')]
        unfrozen_sub_graph.remove_edges_from(to_remove)
        if not os.path.exists('./data/varmisuse/debug'):
            os.makedirs('./data/varmisuse/debug')
        write_dot(unfrozen_sub_graph, os.path.join('./data/varmisuse/debug', str(index) + '.dot'))


def get_shortest_length():
    files_dict = {}
    files_dict['graphs-train'] = os.listdir('./data/varmisuse/graphs-train')
    files_dict['graphs-valid'] = os.listdir('./data/varmisuse/graphs-valid')
    files_dict['graphs-test'] = os.listdir('./data/varmisuse/graphs-test')
    files_dict['graphs-testonly'] = os.listdir('./data/varmisuse/graphs-testonly')
    no_subtoken = False
    no_write_dot = True
    for key in files_dict.keys():
        for file in files_dict[key]:
            samples = load_jsonl_gz(os.path.join('./data/varmisuse', key, file))
            results = parallel_process(samples, single_instance_get_shortest_length, key + ':' + file,
                                       args=(no_subtoken, no_write_dot), n_cores=None)
            save_jsonl_gz(results, os.path.join('./data/varmisuse', key, file))
            gc.collect()


def parallel_process(array, func, file, args=(), n_cores=None):
    if n_cores is 1:
        return [func(x, *args) for x in tqdm(array, desc=file)]
    with tqdm(total=len(array), desc=file) as pbar:
        def update(*args):
            pbar.update()
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=n_cores) as pool:
            jobs = [
                pool.apply_async(func, (x, *args), callback=update) for x in array
            ]
            results = [job.get() for job in jobs]
        return results


def single_instance_get_shortest_length(sample, no_subtoken, no_write_dot):
    index = sample['index']
    if not no_subtoken:
        build_subtoken_edge(sample)
    context_graph = sample['ContextGraph']
    symbol_cands = sample['SymbolCandidates']
    slot_dummy_node = sample['SlotDummyNode']
    cur_graph, symbol_dummy_nodes_correct = build_networkx_graph(index, context_graph, symbol_cands, slot_dummy_node)
    if not os.path.exists('./data/varmisuse/dots'):
        os.makedirs('./data/varmisuse/dots')
    if not no_write_dot:
        write_dot(cur_graph, os.path.join('./data/varmisuse/dots', str(index) + '.dot'))
    cur_graph_undirected = cur_graph.to_undirected()
    for symbol_cand in symbol_cands:
        if symbol_dummy_nodes_correct[symbol_cand['SymbolDummyNode']]:
            if nx.has_path(cur_graph_undirected, slot_dummy_node, symbol_cand['SymbolDummyNode']):
                slot_tgt_line = nx.shortest_path(cur_graph_undirected, source=slot_dummy_node,
                                                 target=symbol_cand['SymbolDummyNode'])
                slot_tgt_line_len = len(slot_tgt_line) - 1
                symbol_cand['Path'] = slot_tgt_line
                symbol_cand['PathLength'] = slot_tgt_line_len
            else:
                symbol_cand['Path'] = []
                symbol_cand['PathLength'] = 0
        else:
            if nx.has_path(cur_graph_undirected, slot_dummy_node, symbol_cand['SymbolDummyNode']):
                slot_cand_line = nx.shortest_path(cur_graph_undirected, source=slot_dummy_node,
                                                  target=symbol_cand['SymbolDummyNode'])
                slot_cand_line_len = len(slot_cand_line) - 1
                symbol_cand['Path'] = slot_cand_line
                symbol_cand['PathLength'] = slot_cand_line_len
            else:
                symbol_cand['Path'] = []
                symbol_cand['PathLength'] = 0
    del cur_graph, cur_graph_undirected
    return sample


def build_subtoken_edge(sample):
    unsplittable_keywords = get_language_keywords('csharp')
    _add_per_subtoken_nodes(unsplittable_keywords, sample['ContextGraph'])


def filter_on_path_length(bins=[1, 3, 5, 6, 7, 8, 10]):
    files_dict = {}
    files_dict['graphs-test'] = os.listdir('./data/varmisuse/graphs-test')
    files_dict['graphs-testonly'] = os.listdir('./data/varmisuse/graphs-testonly')
    for key in files_dict.keys():
        all_filter_results = {}
        all_filter_count = {}
        for i in range(len(bins)):  # the last index is unreachable
            all_filter_count[i] = 0
        for file in files_dict[key]:
            for i in range(len(bins)):  # the last index is unreachable
                all_filter_results[i] = []
            samples = load_jsonl_gz(os.path.join('./data/varmisuse', key, file))
            results = parallel_process(samples, single_instance_filter_on_path_length, key + ':' + file,
                                       args=(bins, ), n_cores=None)
            for result in results:
                for index in result.keys():
                    if bool(result[index]):
                        all_filter_results[index].extend(result[index])
            for index in all_filter_results.keys():
                all_filter_count[index] += len(all_filter_results[index])
                if index == len(bins) - 1:
                    if not os.path.exists(
                            os.path.join('./data/varmisuse/threshold_unreachable/%s' % key)):
                        os.makedirs(
                            os.path.join('./data/varmisuse/threshold_unreachable/%s' % key))
                    save_jsonl_gz(all_filter_results[index],
                                  os.path.join('./data/varmisuse/threshold_unreachable/%s' % key, file))
                    print('%s threshold_%s has %d remaining samples' % (file, 'unreachable', len(all_filter_results[index])))
                else:
                    if not os.path.exists(os.path.join('./data/varmisuse/threshold_%s_%s/%s' % (str(bins[index]), str(bins[index + 1]), key))):
                        os.makedirs(os.path.join('./data/varmisuse/threshold_%s_%s/%s' % (str(bins[index]), str(bins[index + 1]), key)))
                    save_jsonl_gz(all_filter_results[index],
                                  os.path.join('./data/varmisuse/threshold_%s_%s/%s' % (str(bins[index]), str(bins[index + 1]), key), file))
                    print('%s threshold_%s_%s has %d remaining samples' % (file, str(bins[index]), str(bins[index + 1]),
                                                                           len(all_filter_results[index])))
        for index in all_filter_count.keys():
            if index == len(bins) - 1:
                print('Totally %s threshold_%s has %d remaining samples' % (key, 'unreachable', all_filter_count[index]))
            else:
                print('Totally %s threshold_%s_%s has %d remaining samples' % (key, str(bins[index]), str(bins[index + 1]),
                                                                               all_filter_count[index]))


def single_instance_filter_on_path_length(sample, bins):
    path_lengths = []
    symbol_candidates = sample['SymbolCandidates']
    minimum_value = math.inf
    records = {}
    for i in range(len(bins)):                  # the last index is unreachable
        records[i] = []
    for symbol_candidate in symbol_candidates:
        path_length = symbol_candidate['PathLength']
        is_correct = symbol_candidate['IsCorrect']
        if 0 < path_length < minimum_value:     # 0 means infinite
            minimum_value = path_length
        path_lengths.append(path_length)
    if minimum_value == math.inf:
        records[len(bins) - 1].append(sample)
    else:
        index = minimum_in_range(minimum_value, bins)
        flag = True
        for path in path_lengths:
            if index == 0:
                if bins[index] <= path <= bins[index + 1]:
                    continue
                else:
                    flag = False
                    break
            else:
                if bins[index] < path or path == 0:
                    continue
                else:
                    flag = False
                    break
        if flag:
            records[index].append(sample)
    return records


def minimum_in_range(minimum_value, bins):
    for index in range(len(bins)):
        if minimum_value == 1:
            return 0
        elif bins[index] < minimum_value <= bins[index + 1]:
            return index
        else:
            continue


def statistics():
    files_dict = {}
    files_dict['graphs-train'] = os.listdir('./data/varmisuse/graphs-train')
    files_dict['graphs-valid'] = os.listdir('./data/varmisuse/graphs-valid')
    files_dict['graphs-test'] = os.listdir('./data/varmisuse/graphs-test')
    files_dict['graphs-testonly'] = os.listdir('./data/varmisuse/graphs-testonly')
    lengths = []
    for key in files_dict.keys():
        for file in files_dict[key]:
            samples = load_jsonl_gz(os.path.join('./data/varmisuse', key, file))
            for sample in tqdm(samples, desc=key + ':' + file):
                symbol_candidates = sample['SymbolCandidates']
                for symbol_candidate in symbol_candidates:
                    if symbol_candidate['PathLength'] != 0:
                        lengths.append(symbol_candidate['PathLength'])
    arr = np.array(lengths)
    print("mean length %f, max length %f, min length %f " % (np.mean(arr), np.max(arr), np.min(arr)))


def test():
    files_dict = {}
    files_dict['graphs-test'] = os.listdir('./data/varmisuse/graphs-test')
    files_dict['graphs-testonly'] = os.listdir('./data/varmisuse/graphs-testonly')
    for key in files_dict.keys():
        count_dict = {}
        for x in range(1, 8):
            count_dict[x] = 0
        for file in files_dict[key]:
            samples = load_jsonl_gz(os.path.join('./data/varmisuse', key, file))
            for sample in tqdm(samples, desc=key + ':' + file):
                path_length = []
                symbol_candidates = sample['SymbolCandidates']
                for symbol_candidate in symbol_candidates:
                     path_length.append(symbol_candidate['PathLength'])
                if len(list(set(path_length))) == 1:
                    index = int(path_length[0])
                    if index in count_dict.keys():
                        count_dict[index] += 1
                else:
                    continue
        print(count_dict)


if __name__ == '__main__':
    # get_dataset_num()
    # add_index()
    # get_shortest_length()
    # statistics()
    # filter_on_path_length()
    test()
