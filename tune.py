import dgl
from utils import *
from model import *
import torch.nn as nn
import pickle
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import yaml
import random
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def tune(cfg, graph, model, task_heads, opt, W_down, dis, cluster_id, pos_idx, neg_idx):
    epochs = cfg['tune_epochs']

    device = cfg['gpu']

    mean_agg = MeanAggregator()
    head1 = task_heads[0]
    head2 = task_heads[1]

    if device>=0:
        model = model.to(device)
        dis = dis.to(device)
        W_down = W_down.to(device)
        head1 = head1.to(device)
        head2 = head2.to(device)


    feats = graph.ndata['feat']
    labels = graph.ndata['label']

    loss_fn = nn.BCEWithLogitsLoss()


    best_aucroc = 0.0
    best_aucpr = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.eval()  # Freeze 

        sum_l, sum_l1, sum_l2 = 0, 0, 0

        
        H_init = model(feats)

        H_local = torch.mul(H_init, head1)
        H_cluster = torch.mul(H_init, head2)
        H_local_diff = mean_agg.extract_H_diff(graph, H_local, cluster_id, mode='local')
        H_cluster_diff = mean_agg.extract_H_diff(graph, H_cluster, cluster_id, mode='cluster')

        H_concat = torch.cat((H_local_diff, H_cluster_diff), dim=1)

        H_down = torch.matmul(H_concat, W_down)

        center = H_down[neg_idx].mean(dim=0)

        pos_dis = dis(H_down[pos_idx], center, mode='global')
        neg_dis = dis(H_down[neg_idx], center, mode='global')

        l1 = loss_fn(pos_dis, torch.ones_like(pos_dis))
        l2 = loss_fn(neg_dis, torch.zeros_like(neg_dis))
        loss = l1+l2
        
        sum_l += loss.item()
        sum_l1 += l1.item()
        sum_l2 += l2.item()

        opt.zero_grad()
        loss.backward()
        opt.step()
            

        # eval
        model.eval()

        with torch.no_grad():
            eval_h = model(feats)
            eval_h_local = torch.mul(eval_h, head1)
            eval_h_cluster = torch.mul(H_init, head2)
            eval_h_local_diff = mean_agg.extract_H_diff(graph, eval_h_local, cluster_id, mode='local')
            eval_h_cluster_diff = mean_agg.extract_H_diff(graph, eval_h_cluster, cluster_id, mode='cluster')

            eval_h_concat = torch.cat((eval_h_local_diff, eval_h_cluster_diff), dim=1)

            eval_h_down = torch.matmul(eval_h_concat, W_down)

            eval_center = eval_h_down.mean(dim=0)
            score = dis(eval_h_down, eval_center, mode='global').cpu().numpy()

            aucroc = roc_auc_score(labels.cpu().numpy(), score)
            aucpr = average_precision_score(labels.cpu().numpy(), score)


            if aucroc > best_aucroc:
                best_aucroc = aucroc
                best_aucpr = aucpr
                best_epoch = epoch + 1  

        print("Epoch {} | Loss {:.4f} | l1 {:.4f} | l2 {:.4f} | aucroc {:.4f} | aucpr {:.4f}"
              .format(epoch+1, sum_l, sum_l1, sum_l2, aucroc, aucpr))

    logging.info("dataset: {} | hid_dim: {} | layers: {} |  Best epoch: {} | Best AUC-ROC: {:.4f} | Best AUC-PR: {:.4f}"
                 .format(cfg['dataset'], cfg['hid_dim'], cfg['num_layers'], best_epoch, best_aucroc, best_aucpr))



def main(cfg):
    seed_everything(cfg['seed'])
    dataset = cfg['dataset']
    graph = my_load_data(dataset)

    cluster_id_path = './saved_idx/cluster_id/{}.npy'.format(cfg['dataset'])
    cluster_id = np.load(cluster_id_path)

    # self-loop
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()

    labels = graph.ndata['label']
    feats = graph.ndata['feat']
    in_dim = feats.shape[1]
    
    device = cfg['gpu']
    if device>=0:
        torch.cuda.set_device(device)
        graph = graph.to(device)   

    k = cfg['shot_num']
    indices_of_ones = torch.where(labels==1)[0]
    indices_of_zeros = torch.where(labels==0)[0]
    
    selected_pos_idx = random.sample(indices_of_ones.numpy().tolist(), k=k)
    neg_times = cfg['neg_times']
    selected_neg_idx = random.sample(indices_of_zeros.numpy().tolist(), k=k*neg_times)

    model = GNN(graph, in_dim=in_dim, hid_dim=cfg['hid_dim'], activation=nn.PReLU(), layers=cfg['num_layers'], dropout=cfg['dropout'])
    pretrained_model = torch.load(f'/root/cjy/proG/pre_trained/models/{dataset}.pth')
    model.load_state_dict(pretrained_model)

    task_heads = torch.load(f'/root/cjy/proG/pre_trained/task_heads/{dataset}.pth')

    dis = Discriminator(hid_dim=cfg['hid_dim'])
    pretrained_dis = torch.load(f'/root/cjy/proG/pre_trained/dis/{dataset}.pth')
    dis.load_state_dict(pretrained_dis)
   
    hid_dim = cfg['hid_dim']
    W_down = nn.Parameter(torch.randn(2 * hid_dim, hid_dim), requires_grad=True)
    torch.nn.init.xavier_uniform_(W_down)

    opt_para = list(filter(lambda p: p.requires_grad, dis.parameters())) + [W_down]
    opt = torch.optim.AdamW(opt_para, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    tune(cfg, graph, model, task_heads, opt, W_down, dis, cluster_id, selected_pos_idx, selected_neg_idx)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    parser = argparse.ArgumentParser(description='Train model with different hid_dim values.')
    parser.add_argument('--hid_dim', type=int, default=cfg['hid_dim'], help='Hidden dimension for the model.')
    parser.add_argument('--layers', type=int, default=cfg['num_layers'], help='Hidden dimension for the model.')
    args = parser.parse_args()

    cfg['hid_dim'] = args.hid_dim
    cfg['num_layers'] = args.layers

    print(cfg)
    main(cfg)
    
