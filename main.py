import pickle as pkl
import numpy as np
import torch
import networkx as nx
import pandas as pd
from graph import Graph
from GRAB import GRAB

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE using... {DEVICE}")

#load data & perform GRAB 
def run(path : str, FP: str, seed: str):
    print(f'------- path: {path}, FP: {FP}, seed: {seed} --------')

    seed_feature = pkl.load(open(f'{path}/FP_Seed_0.0.pickle', 'rb'))[FP]
    comp_feature = pkl.load(open(f'{path}/FP_Comp_0.0.pickle', 'rb'))[FP]
    features = np.vstack((seed_feature, comp_feature))

    # https://pmc.ncbi.nlm.nih.gov/articles/PMC11060547/
    # use identity matrix as feature (means no node feature given)
    # features = np.eye(features.shape[0])
    # print('----with no feature----')
    features = torch.tensor(features, dtype=torch.float32, device=DEVICE)


    seed_info = pd.read_csv(f'{path}/Data_Seed_0.0.csv')
    seed_info = seed_info.rename(columns={seed_info.columns[0]: 'idx'})
    comp_info = pd.read_csv(f'{path}/Data_Comp_0.0.csv')
    comp_info = comp_info.rename(columns={comp_info.columns[0]: 'idx'})

    selected_id = []
    with open(f'{path}/selected_ids_{seed}.txt', 'r') as f:
        for line in f:
            selected_id.append(line.strip().replace('p', 't'))
    selected_id = np.array(selected_id)


    id_to_idx = {}
    idx_to_id = {}
    
    for index, row in seed_info.iterrows():
        id_to_idx[row['id']] = int(row['idx']) #id = 't'
        idx_to_id[int(row['idx'])] = row['id']
    cnt = len(seed_info)

    for index, row in comp_info.iterrows():
        id_to_idx[str(row['id'])] = int(row['idx']) + cnt
        idx_to_id[int(row['idx'])+cnt] = str(row['id'])

    # put 49 spys from seed_id to unlabeled for training
    seed_id = np.array(seed_info['id'])
    comp_id = np.array(list(map(str,comp_info['id'])))

    spy = np.array([id_to_idx[id] for id in selected_id])
    positive = np.array([id_to_idx[id] for id in seed_id if id not in selected_id])
    compound = np.array([id_to_idx[id] for id in comp_id])
    unlabeled = np.concatenate((spy,compound))

    #construct graph
    raw_edges = []
    with open(f'{path}/edge_500_0.85_{FP}.txt', 'r') as f:
        for line in f:
            raw_edges.append(list(line.split()))

    edges = [ (id_to_idx[src.replace('p', 't')], id_to_idx[tar.replace('p', 't')]) for src, tar, val in raw_edges if src < tar]
    

    graph = {i:[] for i in range(len(id_to_idx))}
    for src, tar in edges :
        graph[src].append(tar)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    index = np.ones(features.shape[0], dtype=bool)
    nodes = features[index].clone().detach()
    edges = torch.tensor(edges)

    graph = Graph(nodes, edges).to(DEVICE)

    #perform GRAB
    output = GRAB(graph, adj, features, unlabeled, positive, FP, seed) # result probability

    # Compute evaluation metrics
    result = {}
    result['probability'] = {idx_to_id[i]: tuple(row.tolist()) for i, row in enumerate(output)}

    pred = (output[:,1]>0.5).int() # result in 1/0

    # True Positives (TP)
    real_p = np.concatenate((spy, positive))
    FN_indices = torch.nonzero(pred[real_p] == 0, as_tuple=False).squeeze()

    if FN_indices.numel() > 0:
        if FN_indices.ndim == 0:  # 스칼라인 경우 처리
            FN_idx = [real_p[FN_indices.item()]]  # 스칼라를 리스트로 변환
        else:
            FN_idx = real_p[FN_indices].tolist()
        result['FNid'] = [idx_to_id[idx] for idx in FN_idx]
    else:
        result['FNid'] = []

    result['FN'] = len(result['FNid'])

    # False Negatives (FN)
    result['TP'] = (pred[real_p] == 1).sum().item()


    # False Positives (FP)
    FP_indices = torch.nonzero(pred[compound] == 1, as_tuple=False).squeeze()

    if FP_indices.numel() > 0:
        if FP_indices.ndim == 0:  # FP_indices가 스칼라인 경우 처리
            FP_idx = [compound[FP_indices.item()]]  # 스칼라를 리스트로 변환
        else:  # FP_indices가 리스트인 경우
            FP_idx = compound[FP_indices].tolist()  # numpy.ndarray -> list로 변환
        result['FPid'] = [idx_to_id[idx] for idx in FP_idx]
    else:
        result['FPid'] = []

    result['FP'] = len(result['FPid'])


    return result


def main():
    fp = {'ChemNP':['standard','avalon','cdk-substructure','estate','extended','fp2','fp4','graph','hybridization','klekota-roth','maccs','pubchem','rdkit'],
          'Subgraph':['subgraph'],
          'SubgraphFreqInstance':['subgraph']}
    
    for seed in ['0','7','28','256','342']:
        result = {}
        for path in [ 'ChemNP','SubgraphFreqInstance', 'Subgraph']:
            for FP in fp[path]:
                result[path if FP == 'subgraph' else FP] = run(f'/home/snu_seoyoung/snu_seoyoung/PU/{path}/result_{seed}', FP, seed)
        with open(f'/home/snu_seoyoung/snu_seoyoung/PU/result_data_{seed}.pickle', 'wb') as f:
            pkl.dump(result, f)

if __name__ == '__main__':
    main()