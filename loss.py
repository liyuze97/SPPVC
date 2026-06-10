import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedContrastiveLoss(nn.Module):
    def __init__(self, beta=2.0, tau=0.5):
        super(WeightedContrastiveLoss, self).__init__()
        self.tau = tau
        self.beta = beta

    def forward(self, h_views):
        m = len(h_views)
        n, d = h_views[0].shape

        h_views = [F.normalize(h, dim=1) for h in h_views]

        total_loss = 0.0
        num_pairs = 0

        for v in range(m):
            for u in range(m):
                if v == u:
                    continue

                h_v = h_views[v]
                h_u = h_views[u]

                sim_vu = torch.matmul(h_v, h_u.T).clamp(min=1e-6, max=1.0)
                with torch.no_grad():
                    rho = (1.0 - sim_vu).pow(self.beta)

                pos_sim = torch.sum(h_v * h_u, dim=1)
                pos_logits = torch.exp(pos_sim / self.tau)

                denom = torch.zeros(n, device=h_v.device)

                for p in range(m):
                    h_p = h_views[p]
                    sim_vp = torch.matmul(h_v, h_p.T)

                    weighted_sim = rho * (sim_vp / self.tau)
                    logits = torch.exp(weighted_sim)

                    denom += logits.sum(dim=1)

                loss = -torch.log(pos_logits / (denom + 1e-8))
                total_loss += loss.mean()
                num_pairs += 1

        return total_loss / num_pairs
