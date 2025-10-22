# 从源代码生成并发dot
import os, sys
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess
import json
from tqdm import tqdm
import re
from pathlib import Path
from normal1 import process_directory # 处理注释
def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    # parser.add_argument('-i ', '--input', help='The dir path of input', type=str, default='/home/all_data_preprocess/4-parse_res/our_data/vul/')
    # parser.add_argument('-o ', '--output', help='The dir path of output', type=str, default='/home/all_data_preprocess/6-json/our_data/6-our-vul/')
    parser.add_argument('-i ', '--input', help='The dir path of input', type=str, default='/home/fzc/dataset/all_data_preprocess/4-parse_res/our_data/vul/')
    parser.add_argument('-iv ', '--input_vul', help='The dir path of input with vulnerability', type=str, default='/home/fzc/dataset/cope_deeprace/vul/')
    parser.add_argument('-inv ', '--input_novul', help='The dir path of input without vulnerability', type=str, default='/home/fzc/dataset/cope_deeprace/novul/')
    # parser.add_argument('-i', '--input', help='A txt file including all path of targeted src files', type=str,default='/home/fzc/dataset/mytest/0day_normal/')
    # parser.add_argument('-o ', '--output', help='The dir path of output', type=str, default='/home/fzc/dataset/all_data_preprocess/6-json/our_data/6-our-novul/')
    parser.add_argument('-o ', '--output', help='The dir path of output', type=str, default='/home/fzc/dataset/cope_deeprace/2_out_condot/')
    parser.add_argument('-t ', '--type', help='The type of procedures: parse or export', type=str, default='export')
    parser.add_argument('-r ', '--repr', help='The type of representation: pdg or lineinfo_json', type=str, default='lineinfo_json')
    args = parser.parse_args()
    return args
def get_all_folders(tmp_path):
    # 获取路径下所有的文件夹
    return [f.path for f in os.scandir(tmp_path) if f.is_dir()]

def process_path(path, scala_file, output_path):
    print('path:', path)
    if os.path.isdir(path):
        files = glob.glob(path + '/*', recursive=False)
        if len(files) > 0:
            os.makedirs(output_path + path.split('/')[-1], exist_ok=True)
            print("joern" + " --script " + scala_file + " --param codeDir=" + path + ' --param outputDir=' + output_path + path.split('/')[-1])
            os.system("joern" + " --script " + scala_file + " --param codeDir=" + path + ' --param outputDir=' + output_path + path.split('/')[-1])

def main():
    # input_path 是源代码文件夹，所有源代码都放在同一个文件夹下
    # input_path_vul是有漏洞源代码文件夹，所有源代码都放在同一个文件夹下
    # input_path_novul是无漏洞源代码文件夹，所有源代码都放在同一个文件夹下
    # output_path 是输出文件夹，所有输出文件每一个都单独在一个目录下
    
    scala_file ='/home/fzc/workspace/vdgraph/joern-parse/joern-export-demo/export_all_ccpg.sc'
    tmp_path='/home/fzc/dataset/cope_deeprace/1_tmp_deeprace/'
    # 打开并读取 JSON 文件


    # 把input_vul下的所有文件都放到tmp_path下
    # 显示input_vul目录下所有文件
    args = parse_options()
    input_vul = args.input_vul
    input_novul = args.input_novul
    output_path = args.output
    # 检查输入路径和输出路径是否以斜杠 / 结尾。如果不是，它们会将斜杠添加到路径的末尾，以确保路径格式正确。
    if input_vul[-1] == '/':
        input_vul = input_vul
    else:
        input_vul += '/'
        
    if input_novul[-1] == '/':
        input_novul = input_novul
    else:
        input_novul += '/'
        
    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'
    
    if tmp_path[-1] == '/':
        tmp_path = tmp_path
    else:
        tmp_path += '/'
        
    print("input_vul: ", input_vul)
    # 处理注释
    process_directory(input_vul)
    process_directory(input_novul)
    
    all_vul_files = glob.glob(input_vul+'/**', recursive=False)
    all_novul_files = glob.glob(input_novul+'/**', recursive=False)
    # 合并两个列表
    all_files = all_vul_files + all_novul_files
    print("all_files: ", all_files)
    # tqdm遍历all_vul_files
    for file in tqdm(all_vul_files):
        # 读取文件内容
        with open(file, 'r', encoding='ISO-8859-1') as f:
            # 按行读取文件内容
            lines = f.readlines()
            for line in lines:
                #去除那些以#开头且以数字结尾的行
                line = [line for line in line if not re.match(r'# \d+ "<.*>"', line)]
            # 写回到文件
            with open(file, 'w', encoding='ISO-8859-1') as f:
                f.writelines(lines)
     
     # 遍历all_novul_files
    for file in tqdm(all_novul_files):
        # 在tmp_path下创建一个文件夹，文件夹名字是0_file的名字，文件夹下写入这个文件
        # 读取文件内容
        with open(file, 'r', encoding='ISO-8859-1') as f:
            data = f.read()
        # 去掉file.split('/')[-1] 后缀名如.c
        
        directory_path = tmp_path+'0_'+os.path.splitext(file.split('/')[-1])[0]
        # directory_path = directory_path.replace('.c', '')
        # 创建文件夹,不存在也创建
        os.makedirs(directory_path, exist_ok=True)
        # 在文件夹中写入文件
        with open(directory_path+'/'+'0_'+file.split('/')[-1], 'w', encoding='utf-8') as fw:
            fw.write(data)
            
     # 遍历all_vul_files
    for file in tqdm(all_vul_files):
        # 在tmp_path下创建一个文件夹，文件夹名字是0_file的名字，文件夹下写入这个文件
        # 读取文件内容
        with open(file, 'r', encoding='ISO-8859-1') as f:
            data = f.read()
            
        directory_path = tmp_path+'1_'+os.path.splitext(file.split('/')[-1])[0]
        # directory_path = directory_path.replace('.c', '')
        # 创建文件夹,不存在也创建
        os.makedirs(directory_path, exist_ok=True)
        # 在文件夹中写入文件
        with open(directory_path+'/'+'1_'+file.split('/')[-1], 'w', encoding='utf-8') as fw:
            fw.write(data)
            
    # 遍历tmp_path下的所有文件夹
    all_paths = get_all_folders(tmp_path)
    # all_paths = glob.glob(tmp_path+'/**', recursive=False)    
    # 多线程的处理   
    with tqdm(total=len(all_paths), desc="Processing paths") as pbar:
        with Pool() as pool:
            for _ in pool.starmap(process_path, [(path, scala_file, output_path) for path in all_paths]):
                pbar.update()
    # with Pool() as pool:
    #     pool.starmap(process_path, [(path, scala_file, output_path) for path in all_paths])
    # 单线程的处理
    # for path in tqdm(all_paths):
    #     print('path:', path)
    #     # 如果是文件夹
    #     if os.path.isdir(path):
    #         # 获取文件夹下的所有文件
    #         files = glob.glob(path+'/*', recursive=False)
    #         # 如果文件夹下有文件
    #         if len(files) > 0:
                
    #             os.makedirs(output_path+path.split('/')[-1], exist_ok=True)
    #             # path.split('/')[-1]
    #             print("joern"+ " --script " + scala_file + " --param codeDir=" + path + ' --param outputDir=' + output_path+path.split('/')[-1])
    #             os.system("joern"+ " --script " + scala_file + " --param codeDir=" + path + ' --param outputDir=' + output_path+path.split('/')[-1]) #
                
                
                
                # joern_process = subprocess.Popen(["joern"], "--script",scala_file,"--param","codeDir=$path","--param","outputDir=$output_path", shell=True, encoding='utf-8')
                # os.system("joern"+ " --script " + "$scala_file" + codeDir= "pdg" + ' --out $out') # cpg 改成 pdg
           
        
            
    # with open('/home/fzc/dataset/deeprace/deeprace_dataset.json', 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # # 打印前三行数据
    # # 遍历data
    # for item in tqdm(data):
    #     # item['func']去除那些以#开头且以数字结尾的行
    #     item['func'] = [line for line in item['func'] if not re.match(r'# \d+ "<.*>"', line)]
        
    #     print(item)
    # # 写回到文件
    # with open('/home/fzc/dataset/deeprace/deeprace_dataset.json', 'w', encoding='utf-8') as f:
    #     json.dump(data, f, indent=4)
    
    # for item in data[:3]:
    #     print(item)
    # # 筛选project为POSIX_Lock_Primitives的数据
    # data = [item for item in data if item['project'] == 'POSIX_Lock_Primitives']
    # # 打印数据长度
    # print(len(data))
    # # tqdm遍历数据，打印func字段
    # for item in tqdm(data):
    #     print(item['func'])
    
    # 调用规则如下：（输入输出的必须都是一个文件夹）
    # joern --script export_all_ccpg.sc --param codeDir=/home/fzc/workspace/vdgraph/joern-parse/joern-export-demo/example/1 --param outputDir=./output3/

if __name__ == '__main__':
    main()