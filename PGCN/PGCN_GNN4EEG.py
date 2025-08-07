import os

import scipy.io as sio
import torch
# from torch_geometric.nn import ChebConv, global_mean_pool, SAGPooling
from torch.utils.data import DataLoader as dataloader
import torch.optim as optim
# from torch_geometric.utils import add_self_loops, degree
from torch.optim.lr_scheduler import CosineAnnealingLR
import load_data
from utils_loss import Metrics
from node_location import *
from torch.nn import Parameter
from model_PGCN import PGCN
from args import *
from tqdm import tqdm
from loader_gnn4eeg.protocols import *


class NormalDataset(Dataset):
    def __init__(self, data, label, device):
        super(NormalDataset, self).__init__()
        self.data = data
        self.label = label
        self.device = device

    def __getitem__(self, ind):
        X = np.array(self.data[ind])  # (seq_length, feat_dim) array
        Y = np.array(self.label[ind]) # (seq_length, feat_dim) arrays
        return torch.from_numpy(X).to(self.device, dtype=torch.float32), torch.from_numpy(Y).to(self.device, dtype=torch.long)

    def __len__(self,):
        return self.data.shape[0]


def data_prepare(args, train_data, train_label, valid_data=None, valid_label=None, mat_train=None, mat_val=None,
                 num_freq=None):
    label_class = set(train_label)
    assert (len(label_class) == args.n_class)

    train_dataset = NormalDataset(train_data, train_label, device)
    valid_dataset = NormalDataset(valid_data, valid_label, device)

    train_loader = dataloader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = dataloader(valid_dataset, args.batch_size, shuffle=False)

    return train_loader, valid_loader


# -------------------- Training --------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    # Hyperparameters
    n_folds = 10
    num_subs = 123
    batch_size = 256
    lr = 1e-6
    weight_decay = 5e-4
    categories = 9
    epochs = 100

    met_calc = Metrics(num_class=categories)

    adj_matrix = Parameter(torch.FloatTensor(convert_dis_m_FACED(get_ini_dis_m_FACED(), 9))).to(device)

    # 返回节点的绝对坐标
    coordinate_matrix = torch.FloatTensor(return_coordinates()).to(device)

    # Standardize features across all nodes
    # placeholder scaler, fit later per fold

    # CV folds
    results = {'acc': [], 'rec': [], 'prec': [], 'f1': []}
    n_per = num_subs // n_folds

    # data_path = 'data_FACED_GNN4EEG\FACED_dataset_9_labels_5s.mat'
    data_path = os.path.join(os.getcwd(), 'FACED_dataset_9_labels_5s.mat')
    loader = data_FACED('cross_subject', 9, data_path)

    for fold in tqdm(range(n_folds)):

        data_train, label_train, data_val, label_val, train_subject_list = loader.getData(
            n_folds, fold)

        train_loader, valid_loader = data_prepare(
            args, data_train, label_train, data_val, label_val)

        # model, opt, sched
        model = PGCN(args, adj_matrix, coordinate_matrix).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        # training loop
        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0
            correct = 0
            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                out, _, _ = model(data)
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1) == label.view(-1)).sum().item()
            train_acc = correct / len(train_loader.dataset)
            scheduler.step()
            # validation
            model.eval()
            val_loss = 0
            preds, labs = [], []
            with torch.no_grad():
                for data, label in valid_loader:
                    data, label = data.to(device), label.to(device)
                    out, _, _ = model(data)
                    val_loss += criterion(out, label).item() * label.size(0)
                    preds.append(out.argmax(1).cpu())
                    labs.append(label.view(-1).cpu())
            val_loss /= len(valid_loader.dataset)

            preds = torch.cat(preds)
            labs = torch.cat(labs)
            acc, rec, prec, f1 = met_calc.compute_metrics(preds, labs)

            # if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Fold {fold} Epoch {epoch:03d}  TrAcc={train_acc:.3f}  ValLoss={val_loss:.4f}  ValAcc={acc:.3f}  F1={f1:.3f}")

        # final metrics
        results['acc'].append(acc)
        results['rec'].append(rec)
        results['prec'].append(prec)
        results['f1'].append(f1)

    # summary
    print("=== Final CV Results ===")
    for k,v in results.items():
        arr = np.array(v)
        print(f"{k}: {arr.mean():.3f} ± {arr.std():.3f}")
