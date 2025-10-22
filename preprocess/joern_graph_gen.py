# python joern_graph_gen.py --repr
# python joern_graph_gen.py
import os, sys
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess


def get_all_file(path):
    path = path[0]
    file_list = []
    path_list = os.listdir(path)
    for path_tmp in path_list:
        full = path + path_tmp + '/'
        for file in os.listdir(full):
            file_list.append(file)
    return file_list

def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    # parser.add_argument('-i ', '--input', help='The dir path of input', type=str, default='/home/all_data_preprocess/4-parse_res/our_data/vul/')
    # parser.add_argument('-o ', '--output', help='The dir path of output', type=str, default='/home/all_data_preprocess/6-json/our_data/6-our-vul/')
    parser.add_argument('-i ', '--input', help='The dir path of input', type=str, default='/home/fzc/dataset/all_data_preprocess/4-parse_res/our_data/vul/')
    # parser.add_argument('-i', '--input', help='A txt file including all path of targeted src files', type=str,default='/home/fzc/dataset/mytest/0day_normal/')
    # parser.add_argument('-o ', '--output', help='The dir path of output', type=str, default='/home/fzc/dataset/all_data_preprocess/6-json/our_data/6-our-novul/')
    parser.add_argument('-o ', '--output', help='The dir path of output', type=str, default='/home/fzc/dataset/all_data_preprocess/6-pdg/our_data/6-our-vul/')
    parser.add_argument('-t ', '--type', help='The type of procedures: parse or export', type=str, default='export')
    parser.add_argument('-r ', '--repr', help='The type of representation: pdg or lineinfo_json', type=str, default='lineinfo_json')
    args = parser.parse_args()
    return args

def joern_parse(file, outdir):
    record_txt =  os.path.join(outdir,"parse_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch "+record_txt)
    with open(record_txt,'r') as f:
        rec_list = f.readlines()
    name = file.split('/')[-1].split('.')[0]
    if name+'\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ',name)
    out = outdir + name + '.bin'
    if os.path.exists(out):
        return
    os.environ['file'] = str(file)
    os.environ['out'] = str(out) #parse后的文件名与source文件名称一致
    os.system('sh joern-parse $file --language c --out $out')
    with open(record_txt, 'a+') as f:
        f.writelines(name+'\n')

def joern_export(bin, outdir, repr):
    record_txt =  os.path.join(outdir,"export_res.txt")
    if not os.path.exists(record_txt):
        os.system("touch "+record_txt)
    with open(record_txt,'r') as f:
        rec_list = f.readlines()

    name = bin.split('/')[-1].replace(".bin","")
    out = os.path.join(outdir, name)
    if name+'\n' in rec_list:
        print(" ====> has been processed: ", name)
        return
    print(' ----> now processing: ',name)
    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)
    
    if repr == 'pdg':
        os.system('sh /home/fzc/software/joern-1.1.172/joern-cli/joern-export $bin'+ " --repr " + "pdg" + ' --out $out') # cpg 改成 pdg
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-pdg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv "+file_path+' '+out+'.dot')
                    os.system("rm -rf "+out)
                    break
        except:
            pass
    else:
        pwd = os.getcwd()
        if out[-4:] != 'json':
            out += '.json'
        print('bin:',bin) # bin: /home/fzc/dataset/all_data_preprocess/4-parse_res/our_data/vul/0_CVE-2016-3751_Android_9d4853418ab2f754c2b63e091c29c5529b8b86ca_171.bin
        print('out:',out) # out: /home/fzc/dataset/all_data_preprocess/6-json/our_data/6-our-vul/0_CVE-2016-3751_Android_9d4853418ab2f754c2b63e091c29c5529b8b86ca_171.json
        # joern_process = subprocess.Popen(["./joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        # joern_process = subprocess.Popen(["joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        joern_process = subprocess.Popen(["/home/fzc/software/joern-1.1.172/joern-cli/joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        import_cpg_cmd = f"importCpg(\"{bin}\")\r"
        script_path = f"{pwd}/graph-for-funcs.sc"
        print('script_path:',script_path)
        run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{out}\"\r" #json
        cmd = import_cpg_cmd + run_script_cmd
        ret , err = joern_process.communicate(cmd)
        print(ret,err)

    len_outdir = len(glob.glob(outdir + '*'))
    print('--------------> len of outdir ', len_outdir)
    with open(record_txt, 'a+') as f:
        f.writelines(name+'\n')

def main():
    # joern_path = '/home/joern-cli_v1.1.172'
    # joern_path = '/usr/local/bin'
    # joern_path = '/home/fzc/software/joern/joern-cli/'
    # joern_path ='/home/fzc/workspace/devign/joern/joern-cli'
    joern_path ='/home/fzc/software/joern-1.1.172/joern-cli'
    os.chdir(joern_path) # 函数将当前工作目录更改为 Joern 工具的路径
    args = parse_options()
    print(args)
    type = args.type
    repr = args.repr

    input_path = args.input
    output_path = args.output
    # 检查输入路径和输出路径是否以斜杠 / 结尾。如果不是，它们会将斜杠添加到路径的末尾，以确保路径格式正确。
    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'
        
    # 创建一个进程池，其中包含 pool_num 个工作进程。
    pool_num = 12
    pool = Pool(pool_num)

    if type == 'parse':
        # files = get_all_file(input_path)
        files = glob.glob(input_path + '*.c')
        pool.map(partial(joern_parse, outdir = output_path), files)

    elif type == 'export':
        bins = glob.glob(input_path + '*.bin')
        if repr == 'pdg':
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)
            #for bin in bins:    
                #joern_export(bin,outdir=output_path,repr=repr)
        else:
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)

    else:
        print('Type error!')    

if __name__ == '__main__':
    main()