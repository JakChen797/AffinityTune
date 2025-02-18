import dgl
from dgl import metis_partition_assignment
from utils import my_load_data, seed_everything
import numpy as np
import pickle
import argparse
from dgl.data import CoraGraphDataset
import yaml
from sklearn.cluster import KMeans


def sample_idx(cluster_id, num_cluster, neg_ratio):
    neg_idx = []
    pos_idx = []
    for i in range(num_cluster):
        curr_pos = list(np.where(cluster_id == i)[0])
        num_neg = int(neg_ratio * len(curr_pos))

        neg_anchor = np.where(cluster_id != i)[0]
        selected_neg_idx = np.random.choice(neg_anchor, num_neg, replace=False)

        neg_idx.append(list(selected_neg_idx))
        pos_idx.append(curr_pos)

    return (pos_idx, neg_idx)


def save_idx(dataset, idx):
    path = f'./saved_idx/sampled_idx/{dataset}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(idx, f)


def check_idx_pure(labels, sampled_idx):
    pos_idx = sampled_idx[0]
    pures = []
    for idx in pos_idx:
        curr_l = labels[idx]
        elements, counts = curr_l.unique(return_counts=True)
        
        most_ele_idx = counts.argmax()
        most_ele = elements[most_ele_idx]
        most_count = counts[most_ele_idx]
        pure = most_count / len(idx)

        print("most label:{}, pure: {:.4f}".format(most_ele, pure))
        pures.append(pure)
    print("mean:{:.4f} std:{:.4f}".format(np.mean(pures), np.var(pures)))


def main(cfg):
    seed_everything(cfg['seed'])

    dataset = cfg['dataset']
    neg_ratio = cfg['neg_ratio']
    k = cfg['num_cluster']

    # 划分图
    graph = my_load_data(dataset)
    print(graph)
    
    # metis划分
    cluster_id = metis_partition_assignment(graph, k=k)

    # k-means 划分
    # feats = graph.ndata['feat'].numpy()
    # kmeans = KMeans(n_clusters=k)
    # kmeans.fit(feats)

    # cluster_id = kmeans.labels_
    
    # 保存cluster_id
    cluster_id_path = f'./saved_idx/cluster_id/{dataset}.npy'
    cluster_id = cluster_id
    np.save(cluster_id_path, cluster_id)

    # 负采样
    sampled_idx = sample_idx(cluster_id, k, neg_ratio)

    # 检查每个cluster的标签纯净度,这里是原标签不是异常标签

    # dgl_g = CoraGraphDataset()[0]

    # labels = dgl_g.ndata['label']
    # check_idx_pure(labels, sampled_idx)

    # 保存采样结果
    save_idx(dataset, sampled_idx)


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    print(cfg)
    main(cfg)
    