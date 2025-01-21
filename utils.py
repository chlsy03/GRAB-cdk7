import os
import random
import numpy as np

import torch
from torch.nn import Linear
from torch.nn import functional as F

from graph import Graph
from loss_function import Soft_NLL_Loss


def init_weights(m):
    if isinstance(m, Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
            
def seed_everything(seed=73):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
def generate_index_set(seed_num, negative_range, positive_range):
    seed_everything(seed=seed_num)
  
    approved_idx = list(negative_range)
    withdrawn_idx = list(positive_range)
    # select half of withdrawn node as spy node (treat as unlabeled node)
    spy_idx = random.sample(withdrawn_idx, round(len(withdrawn_idx)/2))
    withdrawn_idx = list(set(withdrawn_idx).difference(spy_idx))
    # Make unlabeled node index with half of withdrawn node and whole approved node
    unlabeled_idx = approved_idx + spy_idx
    
    test_spy_idx = random.sample(spy_idx, round(len(spy_idx)/2))
    val_spy_idx = list(set(spy_idx).difference(test_spy_idx))
    tn_idx = list(np.load('/data/project/snu_seoyoung/pu_learning/input/DrugBank_true_negatives_index.npy'))
    test_tn_idx = random.sample(tn_idx, round(len(tn_idx)/2))
    val_tn_idx = list(set(tn_idx).difference(test_tn_idx))
    approved_idx = list(set(approved_idx).difference(tn_idx))
    return torch.tensor(withdrawn_idx), torch.tensor(unlabeled_idx), torch.tensor(val_spy_idx), torch.tensor(val_tn_idx), torch.tensor(test_spy_idx), torch.tensor(test_tn_idx), torch.tensor(approved_idx)


def to_LBP_graph(data, device):
    nodes = data.x
    edge_ls = []
    for i in range(data.edge_index.shape[1]):
        if data.edge_index[0][i] < data.edge_index[1][i]:
            edge_ls.append([int(data.edge_index[0][i]), int(data.edge_index[1][i])])
    graph = Graph(nodes, np.array(edge_ls)).to(device)
    return graph

def get_loss(output, beliefs, p_index, u_index, device, class_prior=None):
    output = torch.clamp(output, min=1e-7, max=1 - 1e-7)  # Clamp output values
    u_loss_fn = Soft_NLL_Loss()
    u_output = output[u_index]
    u_belief = beliefs[u_index]

    u_loss = u_loss_fn(u_output, u_belief)
    
    p_output = output[p_index]
    p_output = p_output[:, 1]

    p_loss = torch.mean(-torch.log(p_output))
    if class_prior != None:
        loss = (class_prior*p_loss) + ((1-class_prior)*u_loss)
    else:
        loss = p_loss + u_loss

    return loss



def calculate_new_prior(model, data, u_node):
    model.eval() 
    x = data.x  # Node features
    edge_index = data.edge_index  # Edge indices
    output = model(x, edge_index)[u_node].detach().cpu()
    output_softmax = F.softmax(output, dim=1)

    pred = output_softmax[:, 1] > 0.5
    return sum(pred.numpy()) / len(u_node)