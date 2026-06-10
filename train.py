import torch
import torch.nn.functional as F
import numpy as np
import gc
from loss import WeightedContrastiveLoss
from sklearn.cluster import KMeans
from stochastic import wasserstein_initialisation, regularise_and_invert
from utils import eval_aligned_detail, evaluation
from calculate_graph import calculate_graphs, calculate_laplacian, calculate_cosine_similarity


def pretrain(model, optimizer, config, x1, x2, A1, A2, labels, aligned_idx, unaligned_idx, device):
    for epoch in range(config['pretrain_epoch']+1):
        x1_hat, z1 = model.encoders[0](x1, A1)
        x2_hat, z2 = model.encoders[1](x2, A2)

        # Reconstruction loss
        recon_loss_x = F.mse_loss(x1_hat, x1, reduction='mean') + F.mse_loss(x2_hat, x2, reduction='mean')
        loss = recon_loss_x

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(model, optimizer, x1, x2, A1, A2, labels_0, labels_1, aligned_idx, unaligned_idx, seed, config, device):
    interval = config['got'].get('interval', 100)
    loss_fn = torch.nn.MSELoss()
    acc_list, nmi_list, ari_list, loss_list = [], [], [], []
    criterion = WeightedContrastiveLoss(config['beta'])
    lambda1 = config['lambda_cons']
    P_global = torch.eye(config['num_sample'], device=device)

    for epoch in range(config['train_epoch']+1):
        x1_hat, z1 = model.encoders[0](x1, A1)
        x2_hat, z2 = model.encoders[1](x2, A2)

        h1 = model.feature_contrastive_module(z1)
        h2 = model.feature_contrastive_module(z2)

        # Reconstruction loss
        recon_loss_x = loss_fn(x1_hat, x1) + loss_fn(x2_hat, x2)
        # Contrastive loss
        contrast_loss = criterion([h1[aligned_idx], h2[aligned_idx]])

        total_loss = recon_loss_x + contrast_loss * lambda1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if epoch % config['print_epoch'] == 0:
                h_all = torch.cat([h1, P_global @ h2], dim=1).cpu().numpy()
                y_true = labels_0

                kmeans = KMeans(n_clusters=config['num_class'], n_init=10, random_state=42)
                y_pred = kmeans.fit_predict(h_all)

                acc, nmi, ari = evaluation(y_true, y_pred)

                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)
                print("epoch: {}, loss: {:.4f} ACC :{:.4f} NMI: {:.4f} ARI: {:.4f}".format(epoch, total_loss, acc, nmi, ari))

        if epoch % interval == 0 and epoch not in {0, config['train_epoch']} and config['aligned_ratio'] != 1 and epoch >= config['warmup_epoch']:
            print(f"\n[Epoch {epoch}] Running GOT...")
            mean_init = calculate_cosine_similarity(h1[unaligned_idx].cpu().detach().numpy(),
                                                    h2[unaligned_idx].cpu().detach().numpy())

            mean_init = torch.from_numpy(mean_init).to(device).float()
            model.got.init_param(mean_init)

            g1, g2, L1_reg, L2_reg = get_got_input(h1[unaligned_idx].cpu().detach().numpy(),
                                                   h2[unaligned_idx].cpu().detach().numpy(), config, k=20)
            P_pred = train_got(model.got, L1_reg, L2_reg, optimizer, config, 100, unaligned_idx, labels_0, labels_1, device)
            P_global = build_global_P(P_pred, aligned_idx, unaligned_idx, device=device)

    num = 1
    acc_list = np.array(acc_list)
    idx = np.argsort(-acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)

    best_acc = acc_list[idx[:num]].tolist()
    best_nmi = nmi_list[idx[:num]].tolist()
    best_ari = ari_list[idx[:num]].tolist()

    print("ACC {:.2f} NMI {:.2f} ARI {:.2f}\n".format(np.mean(best_acc) * 100, np.mean(best_nmi) * 100, np.mean(best_ari) * 100))

    acc_list, nmi_list, ari_list, loss_list = [], [], [], []
    # full samples fine-tuning
    for epoch in range(config['finetune_epoch']+1):
        x1_hat, z1 = model.encoders[0](x1, A1)
        x2_hat, z2 = model.encoders[1](x2, A2)

        h1 = model.feature_contrastive_module(z1)
        h2 = model.feature_contrastive_module(z2)

        h2_hat = P_global @ h2
        contrast_loss = criterion([h1, h2_hat], epoch, config)
        total_loss = contrast_loss * lambda1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if epoch % config['print_epoch'] == 0:
                h_all = torch.cat([h1, P_global @ h2], dim=1).cpu().numpy()
                y_true = labels_0

                kmeans = KMeans(n_clusters=config['num_class'], n_init=10, random_state=42)
                y_pred = kmeans.fit_predict(h_all)

                acc, nmi, ari = evaluation(y_true, y_pred)

                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)
                print("epoch: {}, loss: {:.4f} ACC :{:.4f} NMI: {:.4f} ARI: {:.4f}".format(epoch, total_loss, acc, nmi, ari))

    num = 1
    acc_list = np.array(acc_list)
    idx = np.argsort(-acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)

    best_acc = acc_list[idx[:num]].tolist()
    best_nmi = nmi_list[idx[:num]].tolist()
    best_ari = ari_list[idx[:num]].tolist()

    print("ACC {:.2f} NMI {:.2f} ARI {:.2f}".format(np.mean(best_acc) * 100, np.mean(best_nmi) * 100, np.mean(best_ari) * 100))


def train_got(model, L1_reg, L2_reg, optimizer, config, epochs,
              unaligned_idx, labels_0, labels_1, device='cuda', patience=5):
    torch.manual_seed(config['got']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['got']['seed'])

    L1_tensor = torch.from_numpy(L1_reg).float().to(device)
    L2_tensor = torch.from_numpy(L2_reg).float().to(device)

    params = wasserstein_initialisation(L1_reg, L2_reg)

    best_match = -1
    best_P_pred = None
    wait = 0

    for epoch in range(epochs + 1):
        cost = 0.0

        for iter in range(config['got']['num_iter']):
            eps = torch.randn((model.nodes, model.nodes), device=device) * 0.1
            eps = eps - eps.mean() 

            DS = model(eps)

            loss = model.loss_got(L1_tensor, L2_tensor, DS, params)
            cost += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del eps, DS, loss
            torch.cuda.empty_cache()

        cost /= config['got']['num_iter']

        with torch.no_grad():
            mean_copy = model.mean.detach()  
            P = model.doubly_stochastic(mean_copy)
            max_val, max_idx = torch.max(P, dim=1, keepdim=True)
            P_pred = torch.zeros_like(P)
            P_pred.scatter_(1, max_idx, 1)

            label_match = eval_aligned_detail(P_pred, unaligned_idx, labels_0, labels_1)

        if label_match > best_match:
            best_match = label_match
            best_P_pred = P_pred.clone().detach() 
            wait = 0
        else:
            wait += 1

        print('-[Epoch %d] loss: %.4f - Label match: %d / %d - Accuracy: %.4f' %
              (epoch, cost, label_match, len(unaligned_idx), label_match / len(unaligned_idx)))

        del P, max_val, max_idx, P_pred
        torch.cuda.empty_cache()
        gc.collect()

        if wait >= patience:
            break

    print('Best match: %d / %d - Accuracy: %.4f\n' %
          (best_match, len(unaligned_idx), best_match / len(unaligned_idx)))

    return best_P_pred


def get_got_input(fea1, fea2, config, k=100, graph=True):
    g1, g2 = calculate_graphs(fea1, fea2)
    L1 = calculate_laplacian(g1, k=k)
    L2 = calculate_laplacian(g2, k=k)
    if graph:
        [L1_reg, L2_reg] = regularise_and_invert(L1, L2, config['got']['alpha'], ones=True)
    else:
        L1_reg = L1
        L2_reg = L2
    return g1, g2, L1_reg, L2_reg


def build_global_P(P_pred, AlignIndex, UnalignedIndex, device='cuda'):
    N_total = len(AlignIndex) + len(UnalignedIndex)
    P_global = torch.zeros((N_total, N_total), dtype=torch.float32, device=device)

    P_global[AlignIndex, AlignIndex] = 1.0

    idx = torch.tensor(UnalignedIndex, dtype=torch.long, device=device)
    P_global[idx.unsqueeze(1), idx.unsqueeze(0)] = P_pred

    P_global = P_global / (P_global.sum(dim=1, keepdim=True) + 1e-8)

    return P_global





