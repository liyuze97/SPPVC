import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.encoder_gc1 = GraphConvolution(input_dim, hidden_dim)
        self.encoder_gc2 = GraphConvolution(hidden_dim, latent_dim)

        self.feature_decoder = nn.Sequential( nn.Linear(latent_dim, hidden_dim),
                                              nn.ReLU(), nn.Dropout(p=dropout),
                                              nn.Linear(hidden_dim, input_dim) )

    def encode(self, x, adj, dropout=0.2):
        hidden = self.encoder_gc1(x, adj)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, dropout, training=self.training)
        z = self.encoder_gc2(hidden, adj)
        return z

    def decode(self, z):
        x_hat = self.feature_decoder(z)
        return x_hat

    def forward(self, x, adj):
        z = self.encode(x, adj)
        x_hat = self.decode(z)
        return x_hat, z


class GOT(nn.Module):
    def __init__(self, nodes, tau, it):
        super(GOT, self).__init__()
        self.nodes = nodes
        self.tau = tau
        self.it = it

        self.mean = nn.Parameter(torch.zeros((self.nodes, self.nodes)))
        self.log_std = nn.Parameter(torch.full((self.nodes, self.nodes), fill_value=0.0), requires_grad=True)

    def init_param(self, similarity):
        self.mean.data = similarity

    def get_std(self):
        return F.softplus(self.log_std)

    def doubly_stochastic(self, P):
        A = torch.exp(P / self.tau)
        for _ in range(self.it):
            A = A / (A.sum(dim=1, keepdim=True) + 1e-9)
            A = A / (A.sum(dim=0, keepdim=True) + 1e-9)
        return A

    def forward(self, eps):
        std = self.get_std()
        P_noisy = self.mean + std * eps
        DS = self.doubly_stochastic(P_noisy)
        return DS

    def loss_got(self, g1, g2, DS, params):
        [C1_tilde, C2_tilde] = params
        loss_c = torch.trace(g1) + torch.trace(DS @ g2 @ DS.T)

        M = C2_tilde @ DS.T @ C1_tilde
        MMt = M @ M.T
        eigvals = torch.linalg.eigvalsh(MMt)
        sigma = torch.sqrt(torch.clamp(eigvals, min=1e-8))

        loss = loss_c - 2 * torch.sum(sigma)
        return loss



class MyPVC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n = config['num_sample']
        self.num_unaligned = self.n - int(self.n * config['aligned_ratio'])
        self.num_views = 2
        self.shared_dim = config['input_dims'][0][-1]
        self.encoders = nn.ModuleList()

        self.cluster_layer = [Parameter(torch.Tensor(config['num_class'], self.shared_dim)) for _ in range(2)]
        self.cluster_layer.append(Parameter(torch.Tensor(config['num_class'], self.shared_dim)))
        for v in range(3):
            self.register_parameter('centroid_{}'.format(v), self.cluster_layer[v])
        for v in range(2):
            self.encoders.append(GAE(*config['input_dims'][v]))


        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim),
        )
        self.got = GOT(self.num_unaligned, config['got']['tau'], config['got']['it'])

    def predict_distribution(self, z, v, alpha=1.0):
        c = self.cluster_layer[v]
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - c, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def target_distribution(self, q):
        weight = q ** 2/ q.sum(0)
        return (weight.t() / weight.sum(1)).T



