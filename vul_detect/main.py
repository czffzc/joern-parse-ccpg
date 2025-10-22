#  python main.py --task train
#  python main.py --task train --model_type gcn
#  python main.py --task train --model_type gcn --hetero true
import argparse
import os
import pickle
import sys,gc
import joblib,json
import numpy as np
import torch
from torch.nn import BCELoss,CrossEntropyLoss,LogSoftmax,NLLLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import wandb
use_wandb = False
if use_wandb:
    wandb.init(
        project="my-awesome-project-bigvul",  # 项目名称
        name="deepwukong",         # 实验运行的名称（可选）
        config={               # 配置文件（可选）
            "learning_rate": 1e-5,
            "epochs": 32,
            "batch_size": 8
        },
        settings=wandb.Settings(init_timeout=120)
    )
    config = wandb.config
    print(config)

from data_loader.dataset import DataSet, DataSet2
# from model import GGNN_simplify, GCN_simplify2, DevignModel, IVDetect, DeepWukong, RevealModel
from model import  DevignModel, IVDetect, DeepWukong, RevealModel, RGCN, RGAT
from tqdm import tqdm
#from trainer import train, eval
from utils import tally_param, debug

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import torch.nn.functional as F

def save_gru(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    tnum = len(_trainLoader)
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(tqdm(_trainLoader)):
            # if index % 500 == 0:
            #     print(correct)
            #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
            if device != 'cpu':
                data = data.cuda()
            target = data.y.long()
            # if int(target)==0:
            #     target = torch.tensor([1.0,0.0], dtype=long).cuda()
            # else:
            #     target = torch.tensor([0.0,1.0], dtype=long).cuda()
            # optimizer.zero_grad()
            try:
                #out = model(data.x.to(torch.float32), data.edge_index, data.batch)
                outputs = model(data.x.to(torch.float32), data.edge_index)
                outputs = outputs.detach().cpu().tolist()
                edge_index = data.edge_index.detach().cpu().tolist()
                target = int(data.y)
                json_dict = {
                'node_features':outputs,
                'graph':edge_index,
                'target':target}
                # file_path = os.path.join('/home/GNNLRP_model/ivdetect_test', data.name[0])
                file_path = os.path.join('/home/GNNLRP_model/ivdetect_test', data.name[0])
                with open(file_path, 'w', encoding='utf-8') as fp:
                    json.dump(json_dict, fp)
            except:
                tnum-=1
                continue
            

def train(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    tnum = len(_trainLoader)
    print('tnum:',tnum)
    model.train()
    model.zero_grad()
    for index, data in enumerate(tqdm(_trainLoader)):
        # print('data:',data)
        # if index % 500 == 0:
        #     print(correct)
        #     print("curr: {}".format(index) + " train loss: {}".format(train_loss / (index + 1)) + " acc:{}".format(correct / (index + 1)))
        if device != 'cpu':
            data = data.cuda()
            # print('data:',data)
        target = data.y.long()
        # print('target:',target)
        # if int(target)==0:
        #     target = torch.tensor([1.0,0.0], dtype=long).cuda()
        # else:
        #     target = torch.tensor([0.0,1.0], dtype=long).cuda()
        optimizer.zero_grad()
        
        # 用adam优化器优化模型
        
        try:
            #out = model(data.x.to(torch.float32), data.edge_index, data.batch)
            
            if (args.hetero == 'false'):
                # GCN
                out = model(data.x.to(torch.float32), data.edge_index)
            # print('data:',data,data.edge_attr,data.edge_attr.shape)
            else:
                # RGCN
                out = model(data.x.to(torch.float32), data.edge_index,data.edge_attr.squeeze())
            # out = model(data.x.to(torch.float32), data.edge_index,data.edge_attr.squeeze())
        except Exception as e:
            tnum-=1
            print('有错误啊啊啊啊啊')
            # 打印具体的错误
            print(f"发生错误: {e}")
            # print('data:',data,data.edge_attr,data.edge_attr.shape)
            continue
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        #_, predicted = torch.exp(out).max(1)
        #_, predicted = np.argmax(torch.exp(out).numpy(), axis=-1).tolist()
        correct += predicted.eq(target).sum().item()
        del data.x, data.edge_index, data.y, data, predicted, out
    avg_train_loss = train_loss / tnum
    acc = correct / tnum
    print("epochs {}".format(curr_epochs) + " train loss: {}".format(avg_train_loss) + " acc: {}".format(acc))
    gc.collect()
    return avg_train_loss
   
def evaluate_metrics(model, _loader, device):
    print('evaluate >')
    model.eval()
    with torch.no_grad():
        all_predictions, all_targets, all_probs = [], [], []
        print('graph in tqdm(_loader):')
        for graph in tqdm(_loader):
            import time
            # 记录开始时间
            start_time = time.time()
            graph = graph.cuda()
            # print('graph:',graph)
            try:   
                #out = model(graph.x.to(torch.float32), graph.edge_index,graph.batch)
                if args.hetero == 'false':
                    # GCN
                    out = model(graph.x.to(torch.float32), graph.edge_index)
                else:
                    # RGCN
                    out = model(graph.x.to(torch.float32), graph.edge_index,graph.edge_attr.squeeze())
                
                # print('out:',out)
            except:
                continue
            target = graph.y
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            prob_1 = out.cpu().detach().numpy()[0][1]
            # pred = torch.exp(out).argmax(dim=1).cpu().detach().numpy()
            # prob_1 = torch.exp(out).cpu().detach().numpy()[0][1]
            #for index in range(len(target)):
            all_probs.append(prob_1)
            #all_predictions.append(pred[index])
            all_predictions.append(pred)
            #all_targets.append(target[index])
            all_targets.append(target)
            del graph.x, graph.edge_index, graph.y, graph, out
            # 输出一个样本的推理时间
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"推理时间: {inference_time:.4f} 秒")
            
        # print('all_predictions:',all_predictions)
        # print('all_targets:',all_targets)
        # acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        # print(acc)
        # precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        # f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        # recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        # matrix = confusion_matrix(all_targets, all_predictions)
        # #target_names = ['non-vul', 'vul']
        # #report = classification_report(all_targets, all_predictions, target_names=target_names)
        # result = " acc: {}".format(acc) + " precision: {}".format(precision) + " recall: {}".format(recall) + " f1: {}".format(f1) + " \n " 
        # print(result)
        # print('all_targets:',all_targets)
        # print('all_probs:',all_probs)
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        fnr, tnr, _ = roc_curve(all_targets, all_probs, pos_label=0)
        auc_score = round(auc(fpr, tpr) * 100, 2)
        acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        # print('fnr tnr fpr tpr:',fnr,tnr,fpr,tpr)
        # print(acc)
        
        precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        matrix = confusion_matrix(all_targets, all_predictions)
        #target_names = ['non-vul', 'vul']
        #report = classification_report(all_targets, all_predictions, target_names=target_names)
        result = "auc: {}".format(auc_score) + " acc: {}".format(acc) + " precision: {}".format(precision) + " recall: {}".format(recall) + " f1: {}".format(f1) + " \n " 
        # if use_wandb:
        #     wandb.log({"epoch": epoch, "loss": loss,'accuracy': eval_result.accuracy, 'precision': eval_result.precision, 'recall': eval_result.recall, 'f1': eval_result.f1_score, 'fpr': eval_result.fpr, 'fnr': eval_result.fnr})
        print(result)
    model.train()
    return acc,precision,f1,recall,tpr,fpr

def select_accpre(model, _loader, device):
    print('evaluate >')
    model.eval()
    with torch.no_grad():
        for graph in tqdm(_loader):
            graph = graph.cuda()
            try:   
                #out = model(graph.x.to(torch.float32), graph.edge_index,graph.batch)
                out = model(graph.x.to(torch.float32), graph.edge_index)
            except:
                continue
            target = graph.y
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            prob_1 = out.cpu().detach().numpy()[0][1]
            # pred = torch.exp(out).argmax(dim=1).cpu().detach().numpy()
            # prob_1 = torch.exp(out).cpu().detach().numpy()[0][1]
            # if int(target)==1 and int(target)==int(pred):
            if pred == 0:
                with open('/home/my_CodeTransformationTest2/det_res/deepwukong_1.txt','a') as fp:
                    fp.write(graph.name[0]+'\n')
            del graph.x, graph.edge_index, graph.y, graph, out
    model.train()
    return




if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (gcn/ggnn)',
                        choices=['gcn', 'ggnn'], default='gcn')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.',default='devign')
    # parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/fzc/workspace/vdgraph/preprocess/deeprace_preprocess/nvd/')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/fzc/workspace/vdgraph/preprocess/')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')
    parser.add_argument('--subpdg_tag', type=str, help='Name of the node feature.', default='subpdg')
    parser.add_argument('--subpdg_num_tag', type=str, help='Name of the node feature.', default='subpdg_num')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=4)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=1)
    parser.add_argument('--task', type=str, help='train or eval', default='eval')
    parser.add_argument('--hetero', type=str, help='if Heterogeneous graph', default='false')
    args = parser.parse_args()
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    print('model_dir:',model_dir)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    print(args)
    input_dir = args.input_dir
    if args.task != 'eval':
        # processed_data_path = os.path.join('/home/GNNLRP_model/data', 'func_msr_com.bin')
        processed_data_path = os.path.join('/home/fzc/workspace/vdgraph/preprocess', 'cpg_bigvul.bin')
        if True and os.path.exists(processed_data_path):
            print('*'*20)
            dataset_loader = joblib.load(open(processed_data_path, 'rb'))
            #debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
            print('Reading already processed data from %s!' % processed_data_path)
        else:
            print('#'*20)
            print('hhhhhhhhhhhhhhhhhhhh')
            dataset = DataSet2(train_src=os.path.join(input_dir, 'nvd_train.txt'),
                              valid_src=None,
                            #   test_src=os.path.join(input_dir, 'nvd_test.txt'),
                              test_src=os.path.join(input_dir, 'nvd_eval.txt'),
                              )
            # dataset = DataSet(train_src=os.path.join(input_dir, 'nvd_train.txt'),
            #                   valid_src=None,
            #                 #   test_src=os.path.join(input_dir, 'nvd_test.txt'),
            #                   test_src=os.path.join(input_dir, 'nvd_eval.txt'),
            #                   n_ident=args.node_tag,
            #                   g_ident=args.graph_tag,
            #                   l_ident=args.label_tag,
            #                   )
            
            dataset_loader=dataset.dataset_loader
            file = open(processed_data_path, 'wb')
            joblib.dump(dataset_loader, file)
            file.close()
    else:
        # processed_data_path = os.path.join('/home/my_CodeTransformationTest2/det_res', 'test_1.bin')
        processed_data_path = os.path.join('/home/fzc/workspace/vdgraph/preprocess', 'cpg.bin')
        if True and os.path.exists(processed_data_path):
            print('*'*20)
            dataset_loader = joblib.load(open(processed_data_path, 'rb'))
            # debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
            #debug('Reading already processed data from %s!' % processed_data_path)
        else:
            print('#'*20)
            dataset_loader = DataSet2(train_src=None,
                            valid_src=None,
                            test_src=os.path.join(input_dir, 'nvd_eval.txt'),
                            # test_src = '/home/my_CodeTransformationTest2/com_json/1'
                            )
            file = open(processed_data_path, 'wb')
            joblib.dump(dataset_loader, file)
            file.close()

    #assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'


    if args.model_type == 'ggnn':
        model = RevealModel(input_dim=100, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps)
    else:
        model = DeepWukong(input_dim=args.feature_size, output_dim=args.graph_embed_size,
                            )
        # RGCN
        # model = RGCN(input_dim=args.feature_size, output_dim=args.graph_embed_size)
        # RGAT
        # model = RGAT(input_dim=args.feature_size, output_dim=args.feature_size, num_relations=5,heads=4)
    #print('Total Parameters : %d' % tally_param(model))
    print('#' * 100)
    print(model)
    model.cuda()
    #loss_function = BCELoss(reduction='mean')
    loss_function = CrossEntropyLoss()
    #loss_function = NLLLoss()
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    print('batch size  : %d' % args.batch_size)
    print('lr  : 0.0001')
    print('weight_decay  : 0.001')
    # model_dir = '/home/mytest/nvd/only_nvd_output/gnnexp_model/'
    if args.task == 'eval':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # eval(model=model, dataset=dataset_loader, max_steps=2000, dev_every=15,
        #     loss_function=loss_function, optimizer=optim,
        #     save_path=model_dir , max_patience=100, log_every=None) 
        # _save_ckpt_file = torch.load('/home/GNNLRP_model/model_com/reveal/gnn_52.42_82.08_63.91.ckpt')
        _save_ckpt_file = torch.load('/home/fzc/workspace/vdgraph/vul_detect/model/mod_mix59.13_100.0_74.03.ckpt')
        model.load_state_dict(_save_ckpt_file)
        select_accpre(model=model, _loader=dataset_loader['test'], device=device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        starting_epochs = 0
        max_epochs = 100
        # _save_ckpt_file = torch.load('/home/GNNLRP_model/model/IVDetect/gnn_52.33_77.22_62.8.ckpt')
        # model.load_state_dict(_save_ckpt_file)
        # save_gru(1, dataset_loader['test'], model, loss_function, optim , device)
        for e in range(starting_epochs, max_epochs):
            # print(dataset_loader['train'])
            
            # 打印几条dataset_loader['train']的数据
            # for i in range(5):
            #     print(dataset_loader['train'][i])
            train_loss = train(e, dataset_loader['train'], model, loss_function, optim , device)
            
            #os.chdir("/home/IVDetect_MSRdata/result" )
            #torch.save(model, os.getcwd() + "/model/trained_model_{}.pt".format(e))##
            #torch.save(model.state_dict(), "/home/IVDetect_MSRdata/result/model/trained_model_{}.pkl".format(e))
            acc,precision,f1,recall,tpr,fpr = evaluate_metrics(model=model, _loader=dataset_loader['test'], device=device)
            # torch.save(model.state_dict(), '/home/GNNLRP_model/mod_model/ivdetect/'+'mod_mix'+f'{acc}'+'_'+f'{recall}'+'_'+f'{f1}'+'.ckpt')
            if use_wandb:
                wandb.log({"epoch": e, "loss": train_loss,'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr, 'tpr': tpr})
            torch.save(model.state_dict(), '/home/fzc/workspace/vdgraph/vul_detect/model1/'+'mod_mix'+f'{acc}'+'_'+f'{precision}'+'_'+f'{recall}'+'_'+f'{f1}'+'.ckpt')
            # nni.report_intermediate_result(valid_auc)
            if train_loss < 0.3:
                break
            gc.collect()
    # nni.report_final_result(valid_auc)


