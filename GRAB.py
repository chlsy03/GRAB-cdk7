import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from networks import LBP, GCN
from utils import init_weights, get_loss, calculate_new_prior

def GRAB(input_graph,
         input_adj,
         input_features,
         train_u_idx,
         train_p_idx,
         space_name,
         seed_num=73,
         gpu=0):
    print(f'======= Start training: {space_name}_{seed_num} =======')
    ## Set device
    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')

    print(f"device in GRAB using : {device}")

    input_features = input_features.to(device)
    input_shape = input_features.size(1)

    edge_index, edge_attr = dense_to_sparse(torch.tensor(input_adj.todense()))
    data = Data(x=input_features, edge_index=edge_index, edge_attr=edge_attr)
    data = data.to(device)
    
    #initialize models
    lbp = LBP(input_graph, train_p_idx, num_states=2, device=device)

    GCN_model = GCN(input_shape, device).to(device)
    GCN_model.apply(init_weights)
    
    optimizer = torch.optim.Adam(GCN_model.parameters(), lr=5e-3, weight_decay=1e-2)
    for param_group in optimizer.param_groups:
        print(f"Learning Rate: {param_group['lr']}")
        print(f"Weight Decay: {param_group['weight_decay']}")
    
    ## Set learning hyperparameters
    EPSILON = 1e-10
    THRESHOLD = 1e-4
    GCN_epoches = 100
    
    train_count = 0
    old_loss = 1000001.0
    new_loss = 1000000.0
    
    
    print(f'========= Start GRAB training =========')
    #Algorithm 1
    while new_loss - old_loss < 0 or train_count == 0:
        old_loss = new_loss
        if train_count == 0:
            prior = 0
        print('lbp start')
        beliefs = lbp(prior, THRESHOLD, EPSILON)
        print('gcn start')
        new_loss = train(GCN_model, data, beliefs, train_p_idx, train_u_idx, GCN_epoches, optimizer, device)
        prior = calculate_new_prior(GCN_model, data, train_u_idx)   
        log = 'GRAB Epoch: {:03d}, prior: {:.3f}, old loss: {:.4f}, new loss: {:.4f}'
        print(log.format(train_count, prior, old_loss, new_loss))
        print('---------------------------------------------------------------------')
        train_count += 1


    GCN_model.eval()
    x = data.x.to(device)  # Node features
    edge_index = data.edge_index.to(device)  # Edge indices
    output = GCN_model(x, edge_index)
    output = torch.nn.Softmax(dim=1)(output)

    return output #in probability

    
def train(model, data, beliefs, p_node_index, u_node_index, num_epochs, optimizer, device):     
    for GCNepoch in range(num_epochs):
        model.train()
        x = data.x.to(device)  # Node features
        edge_index = data.edge_index.to(device)  # Edge indices
        optimizer.zero_grad()

        output = model(x, edge_index)
        output = torch.nn.Softmax(dim=1)(output)

        loss = get_loss(output, beliefs, p_node_index, u_node_index, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return loss.item()
