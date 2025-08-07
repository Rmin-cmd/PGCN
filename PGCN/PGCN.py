import scipy.io as sio
import torch
# from torch_geometric.nn import ChebConv, global_mean_pool, SAGPooling

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch.optim as optim
# from torch_geometric.utils import add_self_loops, degree
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import load_data
from utils.utils_loss import Metrics
from PAGCN.node_location import *
from torch.nn import Parameter
from model_PGCN import PGCN
from args import *
from tqdm import tqdm


# -------------------- Model --------------------
class GraphDataset(Dataset):
    def __init__(self, features, labels, transform=None, sparsity=0.1):
        super().__init__(None, transform)
        self.features = torch.tensor(features).to(torch.float32)
        self.labels = torch.tensor(labels).to(torch.long)

    def len(self):
        return len(self.labels)

    def get(self, idx):
        x = self.features[idx]
        label = self.labels[idx]

        return x, label


# -------------------- Training --------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    # Hyperparameters
    n_folds = 10
    num_subs = 123
    batch_size = 32
    lr = 1e-7
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

    for fold in tqdm(range(n_folds)):
        # load features & labels
        feat_mat = sio.loadmat(f'../data/features/de_lds_fold{fold}.mat')['de_lds']
        labels_rep = load_data.load_srt_de(feat_mat, True, 'cls9', 11)

        # split
        start = fold * n_per
        end = (fold+1)*n_per if fold < n_folds-1 else num_subs
        val_ids = np.arange(start, end)
        train_ids = np.setdiff1d(np.arange(num_subs), val_ids)

        # reshape data
        X_train = feat_mat[train_ids].reshape(-1, 30, 5)
        X_val   = feat_mat[val_ids].reshape(-1, 30, 5)
        y_train = np.repeat(labels_rep, len(train_ids))
        y_val   = np.repeat(labels_rep, len(val_ids))

        # datasets
        ds_tr = GraphDataset(X_train, y_train)
        ds_va = GraphDataset(X_val,   y_val)
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

        # model, opt, sched
        model = PGCN(args, adj_matrix, coordinate_matrix).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)

        # training loop
        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0
            correct = 0
            for data, label in loader_tr:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                out, _, _ = model(data)
                loss = criterion(out, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1) == label.view(-1)).sum().item()
            train_acc = correct / len(ds_tr)

            # validation
            model.eval()
            val_loss = 0
            preds, labs = [], []
            with torch.no_grad():
                for data, label in loader_va:
                    data, label = data.to(device), label.to(device)
                    out, _, _ = model(data)
                    val_loss += criterion(out, label).item()
                    preds.append(out.argmax(1).cpu())
                    labs.append(label.view(-1).cpu())
            val_loss /= len(loader_va)
            scheduler.step()

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
