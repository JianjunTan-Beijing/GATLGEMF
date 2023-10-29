import os
import os.path as osp
import sys
from openpyxl import load_workbook
import random
import networkx as nx
import gc
import numpy as np
import copy
from sequence_encoder import ProEncoder, RNAEncoder
import csv
from tqdm import tqdm
from util_functions import *
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='node feature')
    # general settings
    parser.add_argument('--dataset', default='NPInter4158', help='network name')
    return parser.parse_args()

args = parse_args()
# default program settings
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"
WINDOW_P_UPLIMIT = 3
WINDOW_P_STRUCT_UPLIMIT = 3
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
CODING_FREQUENCY = True
# get the path
script_dir, script_name = osp.split(os.path.abspath(sys.argv[0]))
parent_dir = osp.dirname(script_dir)

# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/'

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
    ncRNA_serial_number = []
    protein_serial_number = []

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
            ncRNA_serial_number.append(serial_number)
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
            protein_serial_number.append(serial_number)
            serial_number = serial_number + 1
            protein_count = protein_count + 1
        else:
            temp_protein = protein_list[protein_name_index_dict[protein_name]]
        interaction_key = (temp_ncRNA.serial_number, temp_protein.serial_number)
        temp_interaction = ncRNA_Protein_Interaction(temp_ncRNA, temp_protein, label, interaction_key)
        temp_ncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)

        if label == 1:
            interaction_list.append(temp_interaction)
            set_interactionKey.add(interaction_key)
        elif label == 0:
            negative_interaction_list.append(temp_interaction)
            set_negativeInteractionKey.add(interaction_key)
        else:
            print(label)
            raise Exception('{dataset_name}has labels other than 0 and 1'.format(dataset_name=dataset_name))

    print('number of ncRNA：{:d}, number of protein：{:d}, number of node：{:d}'.format(ncRNA_count, protein_count,
                                                                                      ncRNA_count + protein_count))
    print('number of interaction：{:d}'.format(len(interaction_list) + len(negative_interaction_list)))
    return interaction_list, negative_interaction_list, ncRNA_list, protein_list, ncRNA_name_index_dict, protein_name_index_dict,\
    ncRNA_name_serialnumber_dict,protein_name_serialnumber_dict, set_interactionKey, set_negativeInteractionKey,sample_name_serialnumber_dict,\
    ncRNA_serial_number,protein_serial_number

def load_data(data_set):
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_rna_struct.fa')
    pro_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_protein_struct.fa')

    return rna_seqs,pro_seqs ,rna_structs , pro_structs
def read_data_seq(path):
    seq_dict = {}
    seq_serialnumber_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                serialnumber =sample_name_serialnumber_dict[name]
                seq_dict[name] = ''
                seq_serialnumber_dict[serialnumber] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
                    seq_serialnumber_dict[serialnumber] = line
    return  seq_serialnumber_dict

def coding_RNA(ncRNA_serial_number, rna_seqs, rna_structs, RE):
    pbar = tqdm(ncRNA_serial_number,unit = 'iteration')
    RNA_node_feature = {}
    for r in pbar:

    # for r in pairs:
        if r in rna_seqs and r in rna_structs :
            # r_seq = rna_seqs[r]  # rna sequence
            r_struct = rna_structs[r] # rna structure

            # r_conjoint = RE.encode_conjoint(r_seq)
            r_conjoint = RE.encode_conjoint1(r_struct)
            # r_conjoint_struct = RE.encode_conjoint_struct(r_seq, r_struct)
            # #补齐到统一长度
            # supplement = np.zeros(59)  #序列
            supplement = np.zeros(54) #结构
            r_conjoint = np.concatenate((r_conjoint, supplement), axis=0)
            # supplement = np.zeros(113) #序列结构
            # r_conjoint_struct = np.concatenate((r_conjoint_struct, supplement), axis=0) #序列结构
            if r_conjoint is 'Error':
                print('Skip {}  according to conjoint coding process.'.format(r))
            # elif r_conjoint_struct is 'Error':
            #     print('Skip {} according to conjoint_struct coding process.'.format(r))
            else:
                RNA_node_feature[r] = r_conjoint         #r_conjoint_struct
        else:
            print('Skip  {} according to sequence dictionary.'.format(r))
    return RNA_node_feature
def coding_Protein(protein_serial_number, pro_seqs, pro_structs, PE):
    pbar = tqdm(protein_serial_number,unit = 'iteration')
    Protein_node_feature = {}
    for p in pbar:
    # for p in pairs:
        if p in pro_seqs and p in pro_structs:
            # p_seq = pro_seqs[p]  # protein sequence
            p_struct = pro_structs[p]  # protein structure

            # p_conjoint = PE.encode_conjoint(p_seq)
            p_conjoint = PE.encode_conjoint1(p_struct)
            # p_conjoint_struct = PE.encode_conjoint_struct(p_seq, p_struct)
            if p_conjoint is 'Error':
                print('Skip {} according to conjoint coding process.'.format(p))
            # elif p_conjoint_struct is 'Error':
            #     print('Skip {} according to conjoint_struct coding process.'.format(p))

            else:
                Protein_node_feature[p] = p_conjoint            #p_conjoint_struct
        else:
            print('Skip {} according to sequence dictionary.'.format(p))
    return Protein_node_feature
if __name__ == '__main__':
    print('\n' + 'start' + '\n')
    interaction_dataset_path = DATA_BASE_PATH + "pairs/" + args.dataset + '_pairs.xlsx'
    interaction_list, negative_interaction_list, ncRNA_list, protein_list, ncRNA_name_index_dict, protein_name_index_dict, \
    ncRNA_name_serialnumber_dict,protein_name_serialnumber_dict,  set_interactionKey, set_negativeInteractionKey,\
    sample_name_serialnumber_dict,ncRNA_serial_number,protein_serial_number = read_interaction_dataset(dataset_path=interaction_dataset_path,
                                                          dataset_name=args.dataset)
    rna_seqs, pro_seqs, rna_structs, pro_structs = load_data(args.dataset)
    # sequence encoder instances
    RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY)
    PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY)
    #按照serial_number进行编码  ncRNA_name_serialnumber_dict ，protein_name_serialnumber_dict
    RNA_node_feature = coding_RNA(ncRNA_serial_number, rna_seqs, rna_structs, RE)
    Protein_node_feature = coding_Protein(protein_serial_number, pro_seqs, pro_structs, PE)
    #融合RNA特征矩阵和protein特征矩阵，统一到同一维度
    node_feature = {}
    node_feature.update(RNA_node_feature)
    node_feature.update(Protein_node_feature)
    #对node_feature 按照serial_number由小到大排下顺序
    sort_node_feature = {}
    for key in sorted(node_feature.keys()):
        values = node_feature[key]
        sort_node_feature[key] = values

    node_feature = pd.DataFrame(sort_node_feature)
    node_feature = pd.DataFrame(node_feature.values.T)
    node_feature_save_path = RESULT_BASE_PATH +'all_node_information/' + args.dataset +'/nodefeatstruc.csv'
    node_feature.to_csv( node_feature_save_path, index=False)




