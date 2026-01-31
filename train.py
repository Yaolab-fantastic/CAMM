
import torch 
import os
import csv
import random
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from util import *
from model_copy import *
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import time
import psutil

matplotlib.use('Agg')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file = open("/path/to/debug_log.txt", "w")

def log_debug(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)
    return memory_usage
def log_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MB
        return allocated, reserved
    return 0, 0
class DatasetWrapper(Dataset):
    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.classwise_indices_tcp = defaultdict(list)
        self.num_classes = max(self.classwise_indices.keys()) + 1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]
class ClassPairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, minority_classes, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.minority_classes = minority_classes
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for _ in range(len(self)):
            batch_indices = random.sample(indices, self.batch_size)
            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                if y in self.minority_classes:
                    if hasattr(self.dataset, 'classwise_indices_tcp') and \
                       y in self.dataset.classwise_indices_tcp and \
                       len(self.dataset.classwise_indices_tcp[y]) > 0:
                        pair_indices.append(random.choice(self.dataset.classwise_indices_tcp[y]))
                    else:
                        pair_indices.append(random.choice(self.dataset.classwise_indices[y]))
                else:
                    continue 
            yield [idx for idx in batch_indices + pair_indices if idx is not None]

    def __len__(self):
        return self.num_iterations or ((len(self.dataset) + self.batch_size - 1) // self.batch_size)
start_time = time.time()

best_train_acc = 0.0

adj1_dict = load_adj_data('/path/to/data/NCvsAD/AV45', args.adj1)
adj2_dict = load_adj_data('/path/to/data/NCvsAD/FDG/', args.adj2)
adj3_dict = load_adj_data('/path/to/data/NCvsAD/VBM/', args.adj3)
adj_AHBA = np.array(pd.read_csv('/path/to/data/NCvsAD/AHBA.csv', header=None)).astype(float)
adj_AHBA = torch.LongTensor(np.where(adj_AHBA > args.adj_AHBA, 1, 0))

tr_path = '/path/to/data/NCvsAD/X_train.csv' 
te_path = '/path/to/data/NCvsAD/X_test.csv'
tr_df = pd.read_csv(tr_path)
labels = tr_df.iloc[:, -1].values
class_sample_counts = np.bincount(labels)
weights = 1. / class_sample_counts
sample_weights = weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

majority_class = np.argmax(class_sample_counts)
minority_classes = [i for i, c in enumerate(class_sample_counts) if i != majority_class and class_sample_counts[majority_class] / c >= 2]
log_debug(f"[Minority] Classes considered minority: {minority_classes}")

tr_data = CustomDatasetWithAdj(tr_path, adj1_dict, adj2_dict, adj3_dict)
te_data = CustomDatasetWithAdj(te_path, adj1_dict, adj2_dict, adj3_dict)
wrapped_tr_data = DatasetWrapper(tr_data)
te_data_loader = DataLoader(te_data, batch_size=args.batch_size_, shuffle=False)

network = Fusion(num_class=2, num_views=1, hidden_dim=[64], dropout=0.2, in_dim=[116, 116, 116])
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)
class_weights = torch.tensor([class_sample_counts[1] / sum(class_sample_counts), class_sample_counts[0] / sum(class_sample_counts)]).float().to(device)
ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

best_acc = best_te_f1 = best_te_auc = best_te_sen = best_te_spe = 0.0

for epoch in range(2000):
    log_debug(f"Epoch {epoch}/1999\n" + "-" * 10)

    network.eval()
    tcp_vals = {}
    train_loss = 0.0
    train_corrects = 0
    train_num = 0
    tr_probs_all = []
    tr_labels_all = []
    tr_preds_all = []
    memory_before = log_memory_usage()
    gpu_allocated_before, gpu_reserved_before = log_gpu_memory_usage()

    start_inference_time = time.perf_counter()
    
    with torch.no_grad():
            for data in DataLoader(wrapped_tr_data, batch_size=args.batch_size_, shuffle=False):
                sample_id, batch_x, targets, adj1, adj2, adj3 = data
                batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).float().to(device)
                batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).float().to(device)
                batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).float().to(device)
                adj1, adj2, adj3 = adj1.to(device), adj2.to(device), adj3.to(device)
                targets = targets.long().to(device)
                _, _, tr_tcp_vals, *_ = network(batch_x1, batch_x2, batch_x3, adj1, adj2, adj3, adj_AHBA, targets)
                for i, sid in enumerate(sample_id):
                    tcp_vals[sid] = tr_tcp_vals[i].item()
    
    end_inference_time = time.perf_counter()
    inference_time = end_inference_time - start_inference_time
    memory_after = log_memory_usage()
    gpu_allocated_after, gpu_reserved_after = log_gpu_memory_usage()
    log_debug(f"Memory Usage: Before: {memory_before:.2f} MB, After: {memory_after:.2f} MB")
    log_debug(f"GPU Memory Usage: Before: {gpu_allocated_before:.2f} MB (allocated), {gpu_reserved_before:.2f} MB (reserved), "
                f"After: {gpu_allocated_after:.2f} MB (allocated), {gpu_reserved_after:.2f} MB (reserved)")
    log_debug(f"Inference Time: {inference_time:.4f} seconds")
    elapsed_time = time.time() - start_time
    log_debug(f"Elapsed Time: {elapsed_time:.2f} seconds")

    wrapped_tr_data.classwise_indices_tcp = defaultdict(list)
    for i in range(len(wrapped_tr_data)):
        cls = wrapped_tr_data.get_class(i)
        sid = wrapped_tr_data.base_dataset[i][0]
        if sid in tcp_vals and tcp_vals[sid] > 0.5:
            wrapped_tr_data.classwise_indices_tcp[cls].append(i)
    
    for cls in wrapped_tr_data.classwise_indices:
        if len(wrapped_tr_data.classwise_indices_tcp[cls]) == 0:
            wrapped_tr_data.classwise_indices_tcp[cls] = wrapped_tr_data.classwise_indices[cls][:]
    train_loader = DataLoader(
        wrapped_tr_data,
        batch_sampler=ClassPairBatchSampler(wrapped_tr_data, args.batch_size_, minority_classes),
        num_workers=4
    )
    network.train()
    train_corrects = 0
    train_num = 0
    for data in train_loader:
        sample_id, batch_x, targets, adj_matrix1, adj_matrix2, adj_matrix3 = data
        B = batch_x.size(0) // 2
        anchor_x, ref_x = batch_x[:B], batch_x[B:]
        anchor_y, ref_y = targets[:B], targets[B:]
        def to_tensor(x): return x.reshape(-1, 116, 1).float().to(device)
        batch_x1, batch_x2, batch_x3 = to_tensor(anchor_x[:, 0:116]), to_tensor(anchor_x[:, 116:232]), to_tensor(anchor_x[:, 232:])
        adj1, adj2, adj3 = adj_matrix1[:B].to(device), adj_matrix2[:B].to(device), adj_matrix3[:B].to(device)
        anchor_y = anchor_y.long().to(device)

        optimizer.zero_grad()
        loss_fusion, tr_logits, tr_tcp, *_ = network(batch_x1, batch_x2, batch_x3, adj1, adj2, adj3, adj_AHBA, anchor_y)
        loss = loss_fusion

        distill_mask = torch.tensor([y.item() in minority_classes for y in anchor_y], dtype=torch.bool).to(device)
        if distill_mask.any():
            ref_x1, ref_x2, ref_x3 = to_tensor(ref_x[:, 0:116]), to_tensor(ref_x[:, 116:232]), to_tensor(ref_x[:, 232:])
            ref_adj1, ref_adj2, ref_adj3 = adj_matrix1[B:].to(device), adj_matrix2[B:].to(device), adj_matrix3[B:].to(device)
            ref_y = ref_y.long().to(device)

            with torch.no_grad():
                _, ref_logits, *_ = network(ref_x1, ref_x2, ref_x3, ref_adj1, ref_adj2, ref_adj3, adj_AHBA, ref_y)

            T = 4.0
            log_p = F.log_softmax(tr_logits[distill_mask] / T, dim=1)
            q = F.softmax(ref_logits[:log_p.size(0)] / T, dim=1)
            cls_loss = F.kl_div(log_p, q, reduction='batchmean') * (T ** 2)
            if epoch >= 10:  
                loss = loss_fusion + 0.1 * cls_loss  
            else:
                loss = loss_fusion

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * B
        with torch.no_grad():
            tr_prob = F.softmax(tr_logits, dim=1)
            tr_pre_lab = torch.argmax(tr_prob, 1)
        train_corrects += torch.sum(tr_pre_lab == anchor_y.data)
        train_num += B
    train_acc = train_corrects / train_num
    if train_acc > best_train_acc:
        best_train_acc = train_acc

        network.eval()
        all_ids, all_labels = [], []
        gat_embeds, tcp_embeds, distill_embeds = [], [], []

        tr_preds_all.clear()
        with torch.no_grad():
            for data in DataLoader(wrapped_tr_data, batch_size=args.batch_size_, shuffle=False):
                sample_id, batch_x, targets, adj1, adj2, adj3 = data
                batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).float().to(device)
                batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).float().to(device)
                batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).float().to(device)
                adj1, adj2, adj3 = adj1.to(device), adj2.to(device), adj3.to(device)
                targets = targets.long().to(device)

                trr_logits, gat_embed, tcp_embed, distill_embed = network(batch_x1, batch_x2, batch_x3, adj1, adj2, adj3, adj_AHBA, infer=True)
                trr_pre_lab = torch.argmax(F.softmax(trr_logits, dim=1), dim=1)

                gat_embeds.append(gat_embed.cpu())
                tcp_embeds.append(tcp_embed.cpu())
                distill_embeds.append(distill_embed.cpu())
                all_ids.extend(sample_id)
                all_labels.extend(targets.cpu().numpy())
                tr_preds_all.extend(trr_pre_lab.cpu().numpy())

        gat_all = torch.cat(gat_embeds, dim=0).numpy()
        tcp_all = torch.cat(tcp_embeds, dim=0).numpy()
        distill_all = torch.cat(distill_embeds, dim=0).numpy()

        tr_probs_all.extend(tr_prob[:, 1].cpu().numpy())
        tr_labels_all.extend(targets.cpu().numpy())
        tr_preds_all.extend(tr_pre_lab.cpu().numpy())

    network.eval()
    test_corrects = test_num = 0
    te_probs_all, te_labels_all, te_preds_all = [], [], []
    with torch.no_grad():
        for data in te_data_loader:
            sample_id, batch_x, targets, adj1, adj2, adj3 = data
            batch_x1 = batch_x[:, 0:116].reshape(-1, 116, 1).float().to(device)
            batch_x2 = batch_x[:, 116:232].reshape(-1, 116, 1).float().to(device)
            batch_x3 = batch_x[:, 232:].reshape(-1, 116, 1).float().to(device)
            targets = targets.long().to(device)
            adj1, adj2, adj3 = adj1.to(device), adj2.to(device), adj3.to(device)
            te_logits,*_ = network.infer(batch_x1, batch_x2, batch_x3, adj1, adj2, adj3, adj_AHBA)

            if torch.isnan(te_logits).any() or torch.isinf(te_logits).any():
                log_debug(f"[ERROR] te_logits contains NaN/Inf at epoch {epoch}")
                continue

            te_prob = F.softmax(te_logits, dim=1)
            te_pre_lab = torch.argmax(te_prob, 1)

            test_corrects += torch.sum(te_pre_lab == targets.data)
            test_num += batch_x1.size(0)

            te_probs_all.extend(te_prob[:, 1].cpu().numpy())
            te_labels_all.extend(targets.cpu().numpy())
            te_preds_all.extend(te_pre_lab.cpu().numpy())

    te_probs_all_np = np.array(te_probs_all)
    te_labels_all_np = np.array(te_labels_all)
    valid_mask = ~np.isnan(te_probs_all_np)
    te_probs_all_np = te_probs_all_np[valid_mask]
    te_labels_all_np = te_labels_all_np[valid_mask]

    if len(te_labels_all_np) > 0:
        acc = test_corrects.double().item() / test_num
        auc = roc_auc_score(te_labels_all_np, te_probs_all_np)
        f1 = f1_score(te_labels_all_np, te_preds_all)
        sen = sensitivity_score(te_labels_all_np, te_preds_all)
        spe = specificity_score(te_labels_all_np, te_preds_all)

        log_debug(f"[Eval] Epoch {epoch}: acc={acc:.4f}, auc={auc:.4f}, f1={f1:.4f}, sen={sen:.4f}, spe={spe:.4f}")
        log_debug(f"[PredDist] Epoch {epoch}: pred count = {np.bincount(te_preds_all)}")

        if acc > best_acc:
            best_acc, best_te_f1, best_te_auc = acc, f1, auc
            best_te_sen, best_te_spe = sen, spe

log_file.close()
