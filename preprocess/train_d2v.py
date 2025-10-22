from argparse import ArgumentParser
from typing import cast, List
from omegaconf import OmegaConf, DictConfig
import json
import glob
import networkx as nx
# from gensim.models import Doc2Vec, TaggedDocument
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from os import cpu_count
from tqdm import tqdm
from multiprocessing import cpu_count, Manager, Pool
import functools
import os

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
USE_CPU = cpu_count()

def tokenize_code_line(line):
    operators3 = {'<<=', '>>='}
    operators2 = {
        '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
        '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
    operators1 = {
        '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
        '=', ',', '?', ':', ';', '{', '}', '!', '~'
    }

    tmp, w = [], []
    i = 0
    if type(i) == None:
        return []
    while i < len(line):
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

def process_parallel(path: str, split_token: bool):
    node_index = dict()
    tokens_list = list()
    try:
        pdg = nx.drawing.nx_pydot.read_dot(path)
        for index, node in enumerate(pdg.nodes()):
            node_index[node] = index
            try:
                label = pdg.nodes[node]['label'][1:-1]
            except:
                continue
            code = label.partition(',')[2]
            tokens = tokenize_code_line(code)
            tokens_list.append(TaggedDocument(tokens, [index]))
    except:
        pass
    return tokens_list

def train_doc_embedding(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    print('config', config)
    print('config.gnn.embed_size', config.gnn.embed_size)
    cweid = config.dataset.name
    root = config.data_folder
    train_path = "/home/fzc/dataset/all_data_preprocess/6-pdg/our_data/6-our-vul/"
    train_path2 = "/home/fzc/dataset/all_data_preprocess/6-pdg/our_data/6-our-novul/"
    paths2 = glob.glob(train_path2 + '/*')
    paths = glob.glob(train_path + '/*')
    paths = paths + paths2
    print('paths:', paths)
    for dot in tqdm(paths, desc="Processing  :"):
        if os.path.isdir(dot):
            continue
        with open(dot, 'r') as f:
            lines = f.readlines()
        if ':' in lines[0]:
            lines[0] = lines[0].replace(':', '')
        if '~' in lines[0]:
            lines[0] = lines[0].replace('~', '')
            with open(dot, 'w') as f:
                for line in lines:
                    f.write(line)

    documents = list()
    with Manager():
        pool = Pool(8)
        process_func = functools.partial(process_parallel, split_token=config.split_token)
        tokens: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, paths),
                desc=f"pdg paths: ",
                total=len(paths),
            )
        ]
        pool.close()
        pool.join()
    for token_list in tokens:
        documents.extend(token_list)
    print("training doc2vec...")
    print(len(documents))
    num_workers = cpu_count() if config.num_workers == -1 else config.num_workers
    model = Doc2Vec(documents=documents, vector_size=config.gnn.embed_size, window=5, min_count=3, workers=num_workers, seed=64)
    model.save("/home/fzc/dataset/msr/doc2vec_model/doc2vec_pdg_bigvul_nofilter.model")

def load_doc2vec(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name
    model = Doc2Vec.load(f"{config.data_folder}/{cweid}/doc2vec.model")
    print()

if __name__ == '__main__':
    os.chdir("/home/fzc/dataset/mytest/")
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Path to YAML configuration file",
                              default="configs/config.yaml",
                              type=str)
    __args = __arg_parser.parse_args()
    train_doc_embedding(__args.config)
    # load_doc2vec(__args.config)