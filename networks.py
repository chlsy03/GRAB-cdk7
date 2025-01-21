import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from graph import Graph

## Loopy Belief Propagation
class LBP(nn.Module):
    """
    Loopy Belief Propagation.
    """
    def __init__(self,
                 graph: Graph,
                 trn_nodes: Tensor,
                 num_states: int,
                 device: torch.device,
                 epsilon: float = 0.9,
                 diffusion: int = 10):
        """
        Initializer.
        """
        super(LBP, self).__init__()

        self.num_states = 2 # +1 or -1
        self.diffusion = diffusion
        self.epsilon = epsilon
        self.device = device

        self.softmax = nn.Softmax(dim=1)
        self.graph = graph
        self.trn_nodes = trn_nodes
        self.num_edges = self.graph.num_edges()
        self.features = self.graph.get_features()
        self.potential = (
            (torch.ones(self.num_states, device=device) - torch.eye(self.num_states, device=device))*(1.0-self.epsilon) 
            + torch.eye(self.num_states, device=device)*self.epsilon
        )

    def _init_messages(self) -> Tensor:
        """
        Initialize (or create) a message matrix.
        """
        size = (self.num_edges * 2, self.num_states)
        return torch.ones(size, device=self.device) / self.num_states

    def _update_messages(self, messages: Tensor, beliefs: Tensor) -> Tensor:
        """
        Update the message matrix with using beliefs.
        """
        new_beliefs = beliefs[self.graph.src_nodes]
        rev_messages = messages[self.graph.rev_edges]
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential)
        new_msgs = new_msgs / new_msgs.sum(dim=1, keepdim=True)
        return new_msgs

    def _compute_beliefs(self, priors: Tensor, messages: Tensor, EPSILON) -> Tensor:
        """
        Compute new beliefs based on the current messages.
        """
        beliefs = torch.log(torch.clamp(priors, min=EPSILON))
        log_msgs = torch.log(torch.clamp(messages, min=EPSILON))
        beliefs.index_add_(0, self.graph.dst_nodes, log_msgs)
        return self.softmax(beliefs)

    def propagate(self, priors: Tensor, THRESHOLD, EPSILON):
        """
        Propagate the priors produced from the classifier.
        """
        messages = self._init_messages()
        new_messages = messages.clone().detach()
        count = 0
        #Algorithm 2
        while (new_messages - messages).abs().mean().item() > THRESHOLD or count == 0:
            messages = new_messages.clone().detach()
            beliefs = self._compute_beliefs(priors, messages, EPSILON)
            new_messages = self._update_messages(messages, beliefs)
            count += 1
            #print(count, messages, beliefs, new_messages)
            
        beliefs = self._compute_beliefs(priors, new_messages, EPSILON)
        print('number of iteration : ', count)
        return beliefs
    
    def create_node_priors(self, class_prior: float):
        num_nodes = self.num_nodes()
        size = (num_nodes, self.num_states)
        priors = torch.tensor([1.0-class_prior, class_prior], device=self.device, dtype=torch.float32).repeat(num_nodes, 1)
        priors[self.trn_nodes] = torch.tensor([0.0, 1.0], device=self.device)
        return priors

    def num_nodes(self) -> int:
        """
        Count the number of nodes in the current graph.
        """
        return self.graph.num_nodes()
    
    def forward(self, class_prior, THRESHOLD, EPSILON) -> Tensor:
        """
        Run the loopy belief propagation of this model.
        """
        
        priors = self.create_node_priors(class_prior)
        beliefs = self.propagate(priors, THRESHOLD, EPSILON)
        return beliefs

    
class GCN(nn.Module):
    def __init__(self, input_shape, device):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_shape, 16)
        self.conv2 = GCNConv(16, 2)

        self.device = device

    def forward(self, x, edge_index):
        x = x.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def _embedding(self, data):
        x = data.x.clone().detach().to(self.device, dtype=torch.float32)
        edge_index = data.edge_index.clone().detach().to(self.device, dtype=torch.long)

        # First convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        return x
    