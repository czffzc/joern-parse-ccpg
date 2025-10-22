from argparse import ArgumentParser
from typing import cast, List
from omegaconf import OmegaConf, DictConfig
import json
import glob
import networkx as nx
from transformers import RobertaTokenizer, RobertaModel
from os import cpu_count
from tqdm import tqdm
from multiprocessing import cpu_count, Manager, Pool
import functools
import os
import torch

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
USE_CPU = cpu_count()

class CodeBERTEmbedder:
    def __init__(self, model_name='microsoft/codebert-base'):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def embed(self, code_snippet):
        inputs = self.tokenizer(code_snippet, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

def process_parallel(path: str, embedder: CodeBERTEmbedder):
    node_index = dict()
    embeddings_list = list()
    try:
        pdg = nx.drawing.nx_pydot.read_dot(path)
        for index, node in enumerate(pdg.nodes()):
            node_index[node] = index
            try:
                label = pdg.nodes[node]['label'][1:-1]
            except:
                continue
            code = label.partition(',')[2]
            embedding = embedder.embed(code)
            embeddings_list.append(embedding)
    except:
        pass
    return embeddings_list

def train_codebert_embedding(config_path: str):
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

    embedder = CodeBERTEmbedder()
    embeddings_list = list()
    with Manager():
        pool = Pool(8)
        process_func = functools.partial(process_parallel, embedder=embedder)
        embeddings: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, paths),
                desc=f"pdg paths: ",
                total=len(paths),
            )
        ]
        pool.close()
        pool.join()

    print("Embeddings generated...")
    print(len(embeddings))
    print(embeddings)

if __name__ == '__main__':
    os.chdir("/home/fzc/dataset/mytest/")
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Path to YAML configuration file",
                              default="configs/config.yaml",
                              type=str)
    __args = __arg_parser.parse_args()
    train_codebert_embedding(__args.config)