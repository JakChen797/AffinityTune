import dgl
import torch
import torch.nn.functional as F
import random
import os
import dgl.function as fn
from dgl.data.utils import load_graphs
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def t_v_t_split(train_ratio, val_ratio, labels):
    ls = labels.numpy()
    num_nodes = len(ls)

    ano_idx = np.random.permutation(np.nonzero(ls == 1)[0])
    normal_idx = np.random.permutation(np.nonzero(ls == 0)[0])

    train_ano_num = int(train_ratio * len(ano_idx))
    train_normal_num = int(train_ratio * len(normal_idx))

    val_ano_num = int(val_ratio * len(ano_idx))
    val_normal_num = int(val_ratio * len(normal_idx))

    tv_ano_num = train_ano_num + val_ano_num
    tv_normal_num = train_normal_num + val_normal_num

    test_ano_num = len(ano_idx) - tv_ano_num
    test_normal_num = len(normal_idx) - tv_normal_num


    idx_train_ano_all = ano_idx[:train_ano_num]
    idx_train_normal_all = normal_idx[:train_normal_num]

    idx_val = np.concatenate((ano_idx[train_ano_num:tv_ano_num], normal_idx[train_normal_num:tv_normal_num]))

    idx_test = np.concatenate((ano_idx[-test_ano_num:], normal_idx[-test_normal_num:]))

    return idx_train_ano_all, idx_train_normal_all, idx_val, idx_test


def sample_fewshot_ano(idx_train_ano, idx_train_nor, k):
    ano_idx = np.random.permutation(idx_train_ano)
    labeled_idx = ano_idx[:k]
    unlabeled_idx = np.concatenate((idx_train_nor, idx_train_ano[k:]), axis=0)

    return labeled_idx, unlabeled_idx


def idx_shuffle(idxes):
    num_idx = len(idxes)
    idxes = torch.tensor(idxes)
    random_add = torch.randint_like(idxes, high=num_idx, device='cpu')
    idx = torch.arange(0, num_idx)

    shuffled_idx = torch.remainder(idx+random_add, num_idx)

    return shuffled_idx


def row_normalization(feats):
    return F.normalize(feats, p=2, dim=1)


def load_data(dataname, path='./raw_dataset/Flickr'):
    data = sio.loadmat(f'{path}/{dataname}.mat')

    adj = data['Network'].toarray()
    feats = torch.FloatTensor(data['Attributes'].toarray())
    label = torch.LongTensor(data['Label'].reshape(-1))

    graph = dgl.from_scipy(coo_matrix(adj)).remove_self_loop()
    graph.ndata['feat'] = feats
    graph.ndata['label'] = label

    return graph


def my_load_data(dataname, path='../../fullbatchmodel/data/'):
    if dataname in ['Cora', 'Citeseer', 'Pubmed']:
        data_dir = path+dataname+'.bin'
        graph = load_graphs(data_dir)
    elif dataname in ['books', 'enron', 'reddit', 'weibo', 'amazon', 'yelp', 'wiki']:
        graph = load_graphs(f'data/{dataname}.bin')

    return graph[0][0]


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def active_model(model):
    for p in model.parameters():
        p.requires_grad = True


def save_model(model, dis, task_heads, path):
    model_save_path = os.path.join('pre_trained/models', path)
    dis_save_path = os.path.join('pre_trained/dis', path)
    head_tensor = torch.stack(task_heads)
    head_path = os.path.join('pre_trained/task_heads', path)
    torch.save(model.state_dict(), model_save_path)
    torch.save(dis.state_dict(), dis_save_path)
    torch.save(head_tensor, head_path)
    print("Successfully saved model, Discriminator and task heads")


# if __name__ == '__main__':
#     save_model(None, 'cora.pth')