import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import array
import sys, copy, math, time, pdb
import os
import os.path as osp
import random
import argparse
import networkx as nx
from scipy import interp
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net
from itertools import chain
from torch.optim import *
from sklearn import metrics
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve, auc
from openpyxl import load_workbook
import gc
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split,KFold
# from pytorchtools import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description='Link Prediction')
    # general settings
    parser.add_argument('--dataset', default='NPInter4158', help='network name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--initialLearningRate', default=0.0001,type=float, help='Initial learning rate')
    parser.add_argument('--l2WeightDecay', default=0.001, type=float, help='L2 weight')
    parser.add_argument('--epochNumber', default=60,type=int, help='number of training epoch')
    parser.add_argument('--batchSize', default=128, type=int, help='batch size')

    return parser.parse_args()

args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# default program settings
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"
# get the path
script_dir, script_name = osp.split(os.path.abspath(sys.argv[0]))
parent_dir = osp.dirname(script_dir)

# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/'
# set result save path
result_save_path = RESULT_BASE_PATH + args.dataset + "/" + args.dataset + time.strftime(TIME_FORMAT, time.localtime()) + "/"
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)


K_FOLD = 5
stage = 'Line Graph Level Graph Attention Network'
PATIENCES = 10
device = torch.device('cuda:0')

def read_interaction_dataset(dataset_path, dataset_name):
    interaction_list = []
    negative_interaction_list = []
    ncRNA_list = []
    protein_list = []
    ncRNA_name_index_dict = {}
    protein_name_index_dict = {}
    ncRNA_name_serialnumber_dict = {}
    protein_name_serialnumber_dict = {}
    sample_name_serialnumber_dict = {}
    set_interactionKey = set()
    set_negativeInteractionKey = set()
    all_interaction = []
    pos_interaction = []
    neg_interaction = []
    if not osp.exists(dataset_path):
        raise Exception('interaction dataset does not exist')
    print('start reading xlsx file')
    wb = load_workbook(dataset_path)
    sheets = wb.worksheets
    sheet = sheets[0]
    rows = sheet.rows

    serial_number = 0
    ncRNA_count = 0
    protein_count = 0
    flag = 0

    for row in rows:
        if flag == 0:
            flag = flag + 1
            continue

        [ncRNA_name, protein_name, label] = [col.value for col in row]
        label = int(label)
        if ncRNA_name not in ncRNA_name_index_dict:
            temp_ncRNA = ncRNA(ncRNA_name, serial_number, 'ncRNA')
            ncRNA_list.append(temp_ncRNA)
            ncRNA_name_index_dict[ncRNA_name] = ncRNA_count
            ncRNA_name_serialnumber_dict[ncRNA_name] = serial_number
            sample_name_serialnumber_dict[ncRNA_name] = serial_number
            serial_number = serial_number + 1
            ncRNA_count = ncRNA_count + 1
        else:
            temp_ncRNA = ncRNA_list[ncRNA_name_index_dict[ncRNA_name]]
        if protein_name not in protein_name_index_dict:
            temp_protein = Protein(protein_name, serial_number, 'Protein')
            protein_list.append(temp_protein)
            protein_name_index_dict[protein_name] = protein_count
            protein_name_serialnumber_dict[protein_name] = serial_number
            sample_name_serialnumber_dict[protein_name] = serial_number
            serial_number = serial_number + 1
            protein_count = protein_count + 1
        else:
            temp_protein = protein_list[protein_name_index_dict[protein_name]]
        interaction_key = (temp_ncRNA.serial_number, temp_protein.serial_number)
        interaction = (temp_ncRNA.serial_number, temp_protein.serial_number,label)
        all_interaction.append(interaction)
        temp_interaction = ncRNA_Protein_Interaction(temp_ncRNA, temp_protein, label, interaction_key)
        temp_ncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)

        if label == 1:
            interaction_list.append(temp_interaction)
            set_interactionKey.add(interaction_key)
            pos_interaction.append(interaction)
        elif label == 0:
            negative_interaction_list.append(temp_interaction)
            set_negativeInteractionKey.add(interaction_key)
            neg_interaction.add(temp_ncRNA.serial_number, temp_protein.serial_number, label)
        else:
            print(label)
            raise Exception('{dataset_name}has labels other than 0 and 1'.format(dataset_name=dataset_name))

    return interaction_list, negative_interaction_list, ncRNA_list, protein_list, ncRNA_name_index_dict, protein_name_index_dict,\
    ncRNA_name_serialnumber_dict,protein_name_serialnumber_dict, set_interactionKey, set_negativeInteractionKey,sample_name_serialnumber_dict,all_interaction,\
    pos_interaction,neg_interaction

def negative_interaction_generation():
    global ncRNA_list, protein_list, interaction_list, negative_interaction_list, set_interactionKey, set_negativeInteractionKey,all_interaction
    set_negativeInteractionKey = set()

    if len(negative_interaction_list) != 0:
        raise Exception('negative interactions exist')

    num_of_interaction = len(interaction_list)
    num_of_ncRNA = len(ncRNA_list)
    num_of_protein = len(protein_list)

    negative_interaction_count = 0
    while (negative_interaction_count < num_of_interaction):
        random_index_ncRNA = random.randint(0, num_of_ncRNA - 1)  #返回0到num_of_lncRNA - 1之间的任意数
        random_index_protein = random.randint(0, num_of_protein - 1)
        temp_ncRNA = ncRNA_list[random_index_ncRNA]
        temp_protein = protein_list[random_index_protein]
        key_negativeInteraction = (temp_ncRNA.serial_number, temp_protein.serial_number)
        if key_negativeInteraction in set_interactionKey:
            continue
        if key_negativeInteraction in set_negativeInteractionKey:
            continue
        label = 0
        interaction = (temp_ncRNA.serial_number, temp_protein.serial_number,label)
        all_interaction.append(interaction)
        neg_interaction.append(interaction)
        set_negativeInteractionKey.add(key_negativeInteraction)
        temp_interaction = ncRNA_Protein_Interaction(temp_ncRNA, temp_protein, 0, key_negativeInteraction)
        negative_interaction_list.append(temp_interaction)
        temp_ncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)
        negative_interaction_count = negative_interaction_count + 1

def output_information(path:str, information:dict):
    print(f'输出了：{path}')
    with open(path, mode='w') as f:
        information = str(information)
        f.writelines(information)

def get_key(dict,value):
    return [k for k, v in dict.items() if v == value]

def serialnumber_transfer_name(all_interaction):
    all_interaction_name = []
    for i in all_interaction:
        ncRNA_name = get_key(ncRNA_name_serialnumber_dict, i[0])[0]
        protein_name = get_key(protein_name_serialnumber_dict, i[1])[0]
        NPI = (ncRNA_name, protein_name, i[2])
        all_interaction_name.append(NPI)
    return all_interaction_name

def networkx_format_network_generation(interaction_list, ncRNA_list, protein_list):
    edge_list = interaction_list[:]
    node_list = ncRNA_list[:]
    node_list.extend(protein_list)
    node_serialnumber = []
    for node in node_list:
        node_serialnumber.append(node.serial_number)
    sort_node_list = sorted(node_serialnumber)

    G = nx.Graph()
    for node in sort_node_list:
        G.add_node(node)

    for edge in edge_list:
        G.add_edge(edge.ncRNA.serial_number, edge.protein.serial_number)
    print('number of nodes in graph: ', G.number_of_nodes(), 'number of edges in graph: ', G.number_of_edges())
    print(f'number of connected componet : {len(list(nx.connected_components(G)))}')
    result_file.write(f'number of connected componet : {len(list(nx.connected_components(G)))}' + '\n')

    del node_list, edge_list
    gc.collect()
    return G


def output_edgelist_file(G, output_path):
    if not osp.exists(output_path):
        os.makedirs(output_path)
        print(f'创建了文件夹：{output_path}')
    output_path += '/bipartite_graph.edgelist'
    nx.write_edgelist(G, path=output_path)

def get_k_fold_data(k, data):
    data = data.values
    np.random.shuffle(data)
    X, y = data[:, :], data[:, -1]
    sfolder = StratifiedKFold(n_splits = k, shuffle=True)
    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for train, test in sfolder.split(X, y):
        train_data.append(X[train])
        test_data.append(X[test])
        train_label.append(y[train])
        test_label.append(y[test])
    return train_data, test_data

def train(model,device,train_loader,optimizer):
    model.train()
    all_targets = []
    all_scores = []
    all_pred = []
    pbar = tqdm(train_loader, unit='batch')
    train_loss_all = 0
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        logits, loss, pred,prob, _ = model(data)
        all_targets.extend(data.y.tolist())
        all_scores.append(logits[:, 1].cuda().detach())
        all_pred.extend(pred.tolist())
        loss.backward()
        train_loss_all += loss.item()* len(data.y)
        optimizer.step()
    avg_loss = train_loss_all / len(all_targets)
    all_scores = torch.cat(all_scores).cpu().numpy()
    fpr, tpr, thresholds1 = metrics.roc_curve(np.array(all_targets), all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return avg_loss, all_targets, all_pred, auc

def test(model,device,test_loader):
    model.eval()
    all_targets = []
    all_scores = []
    all_pred = []
    test_loss_all = 0
    pbar = tqdm(test_loader, unit='batch')

    with torch.no_grad():
        for data in pbar:
            data = data.to(device)
            all_targets.extend(data.y.tolist())
            logits, loss, pred, prob, _ = model(data)
            all_scores.append(logits[:, 1].cuda().detach())
            all_pred.extend(pred.tolist())
            test_loss_all += loss.item() * len(data.y)
        avg_loss = test_loss_all / len(all_targets)
        all_scores = torch.cat(all_scores).cpu().numpy()

        fpr, tpr, thresholds= metrics.roc_curve(np.array(all_targets), all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    return avg_loss, all_targets, all_pred, all_scores,auc

def performance(tp,tn,fp,fn):
    final_tp = 0
    final_tn = 0
    final_fp = 0
    final_fn = 0
    for i in range(len(tp)):
        final_fn += fn[i]
        final_fp += fp[i]
        final_tn += tn[i]
        final_tp += tp[i]
    print("TN:{},TP:{},FP:{},FN:{}".format(final_tn, final_tp, final_fp, final_fn))
    ACC = (final_tp + final_tn) /float (final_tp + final_tn + final_fn + final_fp)
    Sen = final_tp / float(final_tp+ final_fn)
    Spe = final_tn/float(final_tn+final_fp)
    Pre = final_tp / float(final_tp + final_fp)
    MCC = (final_tp*final_tn-final_fp*final_fn)/float(math.sqrt((final_tp+final_fp)*(final_tn+final_fn)*(final_tp+final_fn)*(final_tn+final_fp)))
    F1 = 2*(Pre*Sen)/(Pre+Sen)
    return ACC,Sen, Spe,Pre,MCC,F1


if __name__ == '__main__':
    start = time.time()
    print("start:{}".format(start))
    result_file = open(result_save_path + 'result.txt', 'w')
    result_file.write(f'database：{args.dataset}\n')
    result_file.write(f'seed = {args.seed}\n')
    result_file.write(f'learn rate：initial = {args.initialLearningRate}，whenever loss increases, multiply by 0.95\n')
    result_file.write(f'L2 weight decay = {args.l2WeightDecay}\n')
    result_file.write(f'number of epoch ：{args.epochNumber}\n')
    result_file.write(f'bachsize ：{args.batchSize}\n')

    interaction_dataset_path = DATA_BASE_PATH + "pairs/" + args.dataset + '_pairs.xlsx'
    interaction_list, negative_interaction_list, ncRNA_list, protein_list, ncRNA_name_index_dict, protein_name_index_dict, ncRNA_name_serialnumber_dict, \
    protein_name_serialnumber_dict, set_interactionKey, set_negativeInteractionKey, sample_name_serialnumber_dict, all_interaction, \
    pos_interaction, neg_interaction = read_interaction_dataset(dataset_path=interaction_dataset_path,
                                                                dataset_name=args.dataset)
    '''
    # 将ncRNA_name_serialnumber_dict，protein_name_serialnumber_dict输出到文件里，方便对应
    all_dict_information = RESULT_BASE_PATH + 'all_dict_information/' + args.dataset
    if not osp.exists(all_dict_information):
        print(f'创建了文件夹：{all_dict_information}')
        os.makedirs(all_dict_information)
    output_information(all_dict_information + '/ncRNA_name_serialnumber_dict.txt', ncRNA_name_serialnumber_dict)
    output_information(all_dict_information + '/protein_name_serialnumber_dict.txt', protein_name_serialnumber_dict)
    '''

    negative_interaction_generation()
    print(
        f'number of ncRNA: {len(ncRNA_list)}, number of protein: {len(protein_list)}')
    print(
        f'number of positive samples: {len(interaction_list)}, number of negative samples: {len(negative_interaction_list)}')
    result_file.write(
        f'number of ncRNA: {len(ncRNA_list)}, number of protein: {len(protein_list)}, number of node: {len(ncRNA_list) + len(protein_list)}'+ '\n')
    result_file.write(
        f'number of positive samples: {len(interaction_list)}, number of negative samples: {len(negative_interaction_list)},number of edges: {len(interaction_list) + len(negative_interaction_list)}' + '\n')
    # 方便比较，将所有相互作用对的serial_number转化为ID保存到某个文件中
    all_interaction_name = serialnumber_transfer_name(all_interaction)
    all_interaction_name = pd.DataFrame(all_interaction_name, columns=['RNA', 'protein', 'label'])
    all_interaction_name.to_csv(RESULT_BASE_PATH + "all_interaction_key/" + args.dataset + '/all_interaction_name10.csv',
                                index=False)

    # 将正样本保存到某一文件中
    # pos_interaction = pd.DataFrame(pos_interaction, columns=['RNA', 'protein', 'label'])
    # pos_interaction.to_csv(RESULT_BASE_PATH + "all_interaction_key/" + args.dataset + '/pos_interaction_key.csv',
    #                        index=False)
    # 将负样本保存到某一文件中
    # neg_interaction = pd.DataFrame(neg_interaction, columns=['RNA', 'protein', 'label'])
    # neg_interaction.to_csv(RESULT_BASE_PATH + "all_interaction_key/" + args.dataset + '/neg_interaction_key.csv',
    #                        index=False)
    # 将正负样本保存到某一文件中
    all_interaction = pd.DataFrame(all_interaction, columns=['RNA', 'protein', 'label'])
    # all_interaction.to_csv(RESULT_BASE_PATH + "all_interaction_key/" + args.dataset + '/all_interaction_key.csv',
    #                        index=False)



    # #read positive pairs and negative pairs
    # all_interaction_key_path = RESULT_BASE_PATH + "all_interaction_key/" + args.dataset
    # all_interaction = pd.read_csv(all_interaction_key_path + '/all_interaction_key.csv')
    # pos_interaction = pd.read_csv(all_interaction_key_path + '/pos_interaction_key.csv')
    # neg_interaction = pd.read_csv(all_interaction_key_path + '/neg_interaction_key.csv')

    train_data, test_data = get_k_fold_data(5, all_interaction)
    G = networkx_format_network_generation(interaction_list, ncRNA_list, protein_list)
    adj = nx.adjacency_matrix(G)
    node_feature = pd.read_csv(osp.join(RESULT_BASE_PATH +'all_node_information/' + args.dataset + '/', 'nodefeat.csv')).values
    # node_feature = None

    print('\n\nK-fold cross validation processes:\n')
    result_file.write(f'{K_FOLD}-fold cross validation processes:\n')

    pred_list = []
    label_list = []
    tps = []
    tns = []
    fns = []
    fps = []

    i = 0
    for fold in range(K_FOLD):
        train_RNA_indices = train_data[fold][:, 0]
        train_protein_indices = train_data[fold][:, 1]
        temp_train_data =(train_RNA_indices,train_protein_indices)
        temp_train_label = train_data[fold][:, 2]
        test_RNA_indices = test_data[fold][:, 0]
        test_protein_indices = test_data[fold][:, 1]
        temp_test_data = (test_RNA_indices, test_protein_indices)
        temp_test_label = test_data[fold][:, 2]
        train_pos_num = len(np.where(train_data[fold][:, 2] == 1)[0])
        train_neg_num = len(np.where(train_data[fold][:, 2] == 0)[0])
        test_pos_num = len(np.where(test_data[fold][:, 2] == 1)[0])
        test_neg_num = len(np.where(test_data[fold][:, 2] == 0)[0])

        A = adj.copy()
        A[temp_test_data[0], temp_test_data[1]] = 0
        A[temp_test_data[1], temp_test_data[0]] = 0
        A.eliminate_zeros()

        train_graphs, test_graphs, max_n_label = links2subgraphs(A, temp_train_data, temp_test_data ,temp_train_label,temp_test_label,node_feature)
        print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))

        print('linegraphs extraction begins.........')
        train_lines = to_linegraphs(train_graphs, max_n_label)
        test_lines = to_linegraphs(test_graphs, max_n_label)

        print(f'##_{fold} FOLD =================================================================')
        result_file.write(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))

        train_loader = DataLoader(train_lines, batch_size=args.batchSize, shuffle=True)
        test_loader = DataLoader(test_lines, batch_size=args.batchSize, shuffle=True)

        # Model configurations
        latent_dim = [32, 32, 32]
        hidden = 128
        num_class = 2
        feat_dim = (max_n_label + 1)*2
        attr_dim = node_feature.shape[1] * 2
        model = Net(feat_dim + attr_dim, hidden, latent_dim, num_class).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.initialLearningRate,weight_decay=args.l2WeightDecay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
        # early_stopping = EarlyStopping(PATIENCES, verbose=False)

        loss_last = float('inf')
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        for epoch in range(args.epochNumber):
            train_loss, train_label, train_pred, train_auc = train(model,device,train_loader,optimizer)
            print(('average train of epoch %d: loss %.5f ' % (epoch, train_loss)))
            train_loss_list.append(train_loss)
            if train_loss > loss_last:
                scheduler.step()
            loss_last = train_loss

            train_pred = [i for k in train_pred for i in k]
            train_pred = torch.tensor(train_pred)
            train_label = torch.tensor(train_label)
            TP, TN, FP, FN = printN(train_pred, train_label)
            train_acc = accuracy(train_pred, train_label)
            train_pre = precision(train_pred, train_label)
            train_sen = sensitivity(train_pred, train_label)
            train_spe = specificity(train_pred, train_label)
            train_MCC = MCC(train_pred, train_label)
            train_F1 = 2 * (train_pre * train_sen) / (train_pre + train_sen)
            train_acc_list.append(train_acc)
            print("ACC:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, F1:{},AUC:{}".
                  format(train_acc, train_pre, train_sen, train_spe, train_MCC, train_F1,train_auc))

            test_loss, test_label, test_pred,test_scores, test_auc = test(model,device,test_loader)
            print(('average test of epoch %d: loss %.5f ' % (epoch, test_loss)))
            test_loss_list.append(test_loss)
            test_pred = [i for k in test_pred for i in k]
            test_pred = torch.tensor(test_pred)
            test_label = torch.tensor(test_label)
            TP, TN, FP, FN = printN(test_pred, test_label)

            test_acc = accuracy(test_pred, test_label)
            test_pre = precision(test_pred, test_label)
            test_sen = sensitivity(test_pred, test_label)
            test_spe = specificity(test_pred, test_label)
            test_MCC = MCC(test_pred, test_label)
            test_F1 = 2 * (test_pre * test_sen) / (test_pre + test_sen)
            test_acc_list.append(test_acc)
            print("ACC:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, F1:{}, AUC:{}".
                    format(test_acc, test_pre, test_sen, test_spe, test_MCC, test_F1,test_auc))

            print("=====================================================")

            if (epoch + 1) ==args.epochNumber:
                tps.append(TP)
                fps.append(FP)
                tns.append(TN)
                fns.append(FN)
                pred_list.append(test_scores)
                label_list.append(test_label)
            # early_stopping(test_loss, model)
            # 若满足 early stopping 要求
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     # 结束模型训练
            #     break
        # # 获得 early stopping 时的模型参数（val_loss下降时所保存的参数）
        # model.load_state_dict(torch.load('checkpoint.pt'))
        result_file.write("ACC:{}, PRE:{}, SEN:{}, SPE:{}, MCC:{}, F1:{}, AUC:{}".
                          format(test_acc, test_pre, test_sen, test_spe, test_MCC, test_F1, test_auc) + '\n')

        network_model_path = result_save_path + 'model fold {}.pt'.format(fold)
        torch.save(model.state_dict(), network_model_path)



    model.train()

    pred_list = np.concatenate(pred_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    # res = {'pred': pred_list ,'label': label_list}
    # res = pd.DataFrame(res)
    # res.to_csv(RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_10.csv', index=False)

    final_AUC = AUC(label_list, pred_list)

    final_ACC, final_Sen, final_Spe, final_Pre, final_MCC,final_F1 = performance(tps, tns, fps, fns)
    print("The final performance of NPI-LGAT is:")
    print("ACC: {}, Sen: {}, Spe: {}, Pre: {}, MCC: {}, F1: {},AUC: {}".format(final_ACC, final_Sen, final_Spe,
                                                                                    final_Pre, final_MCC, final_F1,
                                                                                    final_AUC))
    result_file.write("\n The final performance of NPI-LGAT is:" + '\n')
    result_file.write("ACC: {}, Sen: {}, Spe: {}, Pre: {}, MCC: {}, F1: {},AUC: {}".format(final_ACC, final_Sen, final_Spe,
                                                                                    final_Pre, final_MCC, final_F1,
                                                                                    final_AUC) + '\n')


    end = time.time()
    print("total {} seconds".format(end - start))
    result_file.write('\n' + "total {} seconds".format(end - start))








