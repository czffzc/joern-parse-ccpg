# 从合并的并发dot转化为json
import networkx as nx
from gensim.models import KeyedVectors
import warnings
import argparse
import glob
from multiprocessing import Pool
from functools import partial
import numpy as np
import json
import os
import pydot
warnings.filterwarnings("ignore")
cnt = 0
from tqdm import tqdm
def tokenize_code_line(line):
    # Sets for operators
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
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Check operators and append to final list
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
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    # Filter out irrelevant strings
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

def joern_to_devign(dot_pdg, word_vectors, out_path):
    print('dot_pdg:',dot_pdg)
    print(dot_pdg.split("/")[-2])
    print('out_path:',out_path)
    # out_path = out_path + dot_pdg.split("/")[3]+'/'
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # print('out_path:',out_path)
    # 生成输出 JSON 文件的路径和名称：
    # name = dot_pdg.split('/')[-1].split('.')[0]
    name = dot_pdg.split('/')[-2]
    print('name:',name)
    out_json = out_path + name + '.json'
    # print('out_json:',out_json) 
    if os.path.exists(out_json):
        print("-----> has been processed :\t", out_json)
        return
    print("===============\t"+dot_pdg)
    vul = int(name.split('_')[0])
    node_index = dict()
    node_feature = dict()
    try:
        pdg = nx.drawing.nx_pydot.read_dot(dot_pdg)
        # pdg = nx.drawing.nx_agraph.read_dot(dot_pdg)
        # 读取 .dot 文件
        # (pdg,) = pydot.graph_from_dot_file(dot_pdg)
        # print('pdg:',pdg)
        if type(pdg) != None:
            for index, node in enumerate(pdg.nodes()):
            #     print('index:',index)
            #     print('node:',node)
                node_index[node] = index
                label = pdg.nodes[node]['label'][1:-1]
                code = label.partition(',')[2]
                # print('code:',code)
                feature = np.array([0.0 for i in range(100)])
                for token in tokenize_code_line(code):
                    # print('token:',token)
                    # print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
                    if token in word_vectors:
                        # print('token in word_vectors',token,word_vectors[token],token,word_vectors[token].size)
                        feature += np.array(word_vectors[token])
                        # print('feature',feature)
                    else:
                        feature += np.array([0.0 for i in range(100)])
                node_feature[index] = feature
                # print('node_feature[index]',feature)

            nodes_ = []
            for i in range(len(list(pdg.nodes()))):
                nodes_.append(list(node_feature[i]))

            edges_ = []
            for item in pdg.adj.items():
                print('item:',item)
                s = item[0]
                for edge_relation in item[1]:
                    print('edge_relation:',edge_relation)
                    d = edge_relation    
                    ddg_flag = 0
                    cdg_flag = 0 
                    for edge in item[1]._atlas[edge_relation].items():
                        if 'DDG' in edge[1]['label'] and ddg_flag == 0:
                            edge_type = 0
                            ddg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CDG' in edge[1]['label'] and cdg_flag == 0:
                            edge_type = 1
                            cdg_flag = 1
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        # elif 'AST' in edge[1]['label']:
                        #     edge_type = 4
                        #     edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'CFG' in edge[1]['label']:
                            edge_type = 2
                            edges_.append((node_index[s], edge_type, node_index[d]))
                        elif 'thread' in edge[1]['label']:
                            edge_type = 3
                            edges_.append((node_index[s], edge_type, node_index[d]))
                            
            data = dict()
            data['node_features'] = nodes_
            data['graph'] = edges_
            data['target'] = vul
            out_json = out_path + name + '.json'
            print('out_json',out_json)
            with open(out_json, 'w') as f:
                f.write(json.dumps(data))  

    except:
        # 有错误发生
        # 把错误文件路径输出到log文件中
        with open('error_log.txt', 'a') as f:
            f.write(dot_pdg + '\n')
        print('hhhhhhhhhhhhhhhhhhhhhhhh')
        pass
    return 

# 获取 args.input_dir 下所有的文件夹
def get_folders_in_directory(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir()]
def main():
    '''dir_path_list = ['/home/joern_res_pdg/ffmpeg_novul/', '/home/joern_res_pdg/ffmpeg_vul/',
    '/home/joern_res_pdg/qemu_novul/', '/home/joern_res_pdg/qemu_vul/']
    out_path = '/home/devign_pdg_new/'
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/test1/')
    # parser.add_argument('--output_dir', type=str, help='Output Directory of the parser',default='/home/test2/')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/fzc/dataset/cope_deeprace/2_out_condot/')
    parser.add_argument('--output_dir', type=str, help='Output Directory of the parser',default='/home/fzc/dataset/cope_deeprace/3_out_json_without_ast/')
    args = parser.parse_args()

    # dir_path_list = [args.input_dir]
    dir_path_list = get_folders_in_directory(args.input_dir)
    # dir_path_list为args.input_dir下的所有文件夹的列表
    #   
    # print(dir_path_list)
    out_path = args.output_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dots = []
    for dir_path in dir_path_list:
        # 查找dir_path下所有扩展名为 .dot 的文件，并将这些文件的路径存储在列表 dots_tmp
        # dots_tmp
        # dots_tmp = glob.glob(dir_path + '*.dot') # 查找指定目录中所有扩展名为 .dot 的文件，并将这些文件的路径存储在列表 dots_tmp
        dots_tmp = glob.glob(os.path.join(dir_path, '**', '*.dot'), recursive=True)
        print('dots_tmp:',dots_tmp)
        for dot in dots_tmp:
            if "0_" or 'novul' in dir_path: #无漏洞的为0_开头
                # new_name = '0_'+dir_path+'.dot'
                new_name = dir_path+'/combined_functions.dot'
            else: # 有漏洞的为1_开头
                # new_name = '1_'+dir_path+'.dot'
                new_name = dir_path+'/combined_functions.dot'
            # if '-' in dot.rsplit('/')[-1]: #说明是从joern直接解析的结果，没有重新命名过
            #     if "0_" or 'novul' in dir_path: #无漏洞的为0_开头
            #         new_name = dot[:dot.rindex('/')+1] + '0_' + dot[dot.rindex('/')+1:].replace("-pdg","")
            #     else: # 有漏洞的为1_开头
            #         new_name = dot[:dot.rindex('/')+1] + '1_' + dot[dot.rindex('/')+1:].replace("-pdg","")
            #     os.system("mv "+ dot + ' ' + new_name)
            # else:
            #     new_name = dot
            dots.append(new_name)
    print('dots:',dots)
    # dots是一个文件list,遍历每一个文件，如果第一行内容存在冒号则去掉冒号，否则后面nx.drawing.nx_pydot.read_dot会解析错误
    for dot in tqdm(dots, desc="Processing  :"):
        # 判断是文件夹还是文件
        if os.path.isdir(dot):
            continue
        with open(dot, 'r') as f:
            lines = f.readlines()
        if ':' in lines[0]:
            lines[0] = lines[0].replace(':', '') # 去掉第一行中的所有冒号
        if '~' in lines[0]:
            lines[0] = lines[0].replace('~', '') # 去掉第一行中的所有~
            with open(dot, 'w') as f:
                for line in lines:
                    f.write(line)
                
    #读取词向量模型w2v
    # word_vectors = KeyedVectors.load('/home/mVulPreter/w2v.wv', mmap='r')
    # word_vectors = KeyedVectors.load('/home/fzc/workspace/vdgraph/vul_detect/w2v_complete.wv', mmap='r')
    word_vectors = KeyedVectors.load('/home/fzc/dataset/msr/w2v_model/w2v_pdg_2024_11_9.wv', mmap='r')
    pool = Pool(4)
    pool.map(partial(joern_to_devign, word_vectors=word_vectors, out_path=out_path), dots)

if __name__ == '__main__':
    main()

