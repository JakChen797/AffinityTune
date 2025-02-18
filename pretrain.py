import dgl
from utils import *
from model import *
import torch.nn as nn
import pickle
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import yaml

def sample_negidx4node(cluster_id, num_cluster):
    # sample negative index for each node
    num_node = len(cluster_id)
    node_id = list(range(num_node))

    neg_node_idx = idx_shuffle(node_id)

    neg_cluster_idx = []

    unique_cluster_id = np.arange(num_cluster)
    for x in cluster_id:
        other_cluster = unique_cluster_id[unique_cluster_id!=x]
        sampled_cluster = np.random.choice(other_cluster)
        neg_cluster_idx.append(sampled_cluster)
    return neg_node_idx, neg_cluster_idx
    
    

def train(cfg, graph, model, opt, dis, cluster_id, task_heads, is_save_model):
    epochs = cfg['epochs']
    device = cfg['gpu']
    mean_agg = MeanAggregator()
    # task heads
    head1, head2 = task_heads
    if device>=0:
        model = model.to(device)
        dis = dis.to(device)
        mean_agg = mean_agg.to(device)
        head1 = head1.to(device)
        head2 = head2.to(device)

    feats = graph.ndata['feat']
    labels = graph.ndata['label']
    num_nodes = len(labels)
    node_idx = np.arange(0, num_nodes)
    num_cluster = cfg['num_cluster']

    unique_cluster_id, cluster_count = np.unique(cluster_id, return_counts=True)
    # print(num_cluster)
    # print(unique_cluster_id)
    loss_fn = nn.BCEWithLogitsLoss()

    best = 1e9

    for epoch in range(epochs):
        model.train()
        
        embed = model(feats)
        
        neg_node_idx, neg_cluster_idx = sample_negidx4node(cluster_id, num_cluster)

        # loss func
        local_embed = torch.mul(embed, head1)
        local_mean_h = mean_agg(graph, local_embed)
        local_pos_dis = dis(local_embed, local_mean_h, mode='local')
        local_neg_dis = dis(local_embed, local_mean_h[neg_node_idx], mode='local')

        local_l1 = loss_fn(local_pos_dis, torch.ones_like(local_pos_dis))
        local_l2 = loss_fn(local_neg_dis, torch.zeros_like(local_neg_dis))
        local_loss = local_l1 + local_l2

        
        cluster_embed = torch.mul(embed, head2)
        # cluster centers
        cluster_centers = []
        for cluster in unique_cluster_id:
            curr_feats = cluster_embed[cluster_id == cluster]
            cluster_centers.append(torch.mean(curr_feats, dim=0))

        cluster_centers = torch.stack(cluster_centers) 
        cluster_pos_dis = dis(cluster_embed, cluster_centers[cluster_id], mode='local')
        cluster_neg_dis = dis(cluster_embed, cluster_centers[neg_cluster_idx], mode='local')
        
        cluster_l1 = loss_fn(cluster_pos_dis, torch.ones_like(local_pos_dis))
        cluster_l2 = loss_fn(cluster_neg_dis, torch.zeros_like(local_neg_dis))
        cluster_loss = cluster_l1 + cluster_l2
        
        # sum_l += loss.item()
        # sum_l1 += l1.item()
        # sum_l2 += l2.item()
        loss = local_loss + cluster_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    if is_save_model:
        dataset_name = cfg['dataset']
        path = f'{dataset_name}.pth'
        save_model(model, dis, [head1, head2], path)



def main(cfg):
    seed_everything(cfg['seed'])
    graph = my_load_data(cfg['dataset'])

    # idx_path = './saved_idx/sampled_idx/{}.pkl'.format(cfg['dataset'])
    # with open(idx_path, 'rb') as f:
    #     pos_idx, neg_idx = pickle.load(f)
    cluster_id_path = './saved_idx/cluster_id/{}.npy'.format(cfg['dataset'])
    cluster_id = np.load(cluster_id_path)

    feats = graph.ndata['feat']
    in_dim = feats.shape[1]
    hid_dim = cfg['hid_dim']
    device = cfg['gpu']
    if device>=0:
        torch.cuda.set_device(device)
        graph = graph.to(device)   

    model = GNN(graph, in_dim=in_dim, hid_dim=hid_dim, activation=nn.PReLU(), layers=cfg['num_layers'], dropout=cfg['dropout'])
    dis = Discriminator(hid_dim=hid_dim)

    # task head
    task_head1 = nn.Parameter(torch.rand(hid_dim))
    task_head2 = nn.Parameter(torch.rand(hid_dim))
    task_heads = [task_head1, task_head2]
    opt_para = list(model.parameters()) + list(dis.parameters()) + task_heads

    opt = torch.optim.AdamW(opt_para, lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    train(cfg, graph, model, opt, dis, cluster_id, task_heads, is_save_model=False)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    print(cfg)
    main(cfg)
    