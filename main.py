import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
import torch.nn.functional as F


def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of sequence.
    """
    d = 128
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return np.array(res)


def data_construct(seqs, labels, train):
    # Amino acid dictionary
    '''
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
                'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    '''
    # aa_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    aa_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
               'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21}

    longest_num = len(max(seqs, key=len))
    sequences = [i.ljust(longest_num, 'X') for i in seqs]    # 补长
    pos_embed = position_encoding(sequences)     # 位置嵌入
    HF_feature = HF_encoding(seqs, sequences)    # 传统特征

    pep_codes = []
    for pep in seqs:    # 读每个序列
        current_pep = []
        for aa in pep:    # 读每个残基
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))     # one-hot编码

    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True)  # Fill the sequence to the same length

    dataset = Data.TensorDataset(embed_data, torch.FloatTensor(pos_embed),  # 封装one-hot+位置+传统+标签
                                 torch.FloatTensor(HF_feature), torch.LongTensor(labels))
    batch_size = 15
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_iter


def load_bench_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()    # 数据已有标签
    data_iter = data_construct(seqs, labels, train=True)
    # data_iter = list(data_iter)
    train_iter = [x for i, x in enumerate(data_iter) if i % 5 != 0]   # i是索引，x是内容
    test_iter = [x for i, x in enumerate(data_iter) if i % 5 == 0]    # 5个里面有一个放验证集里，其余为训练数据，训练：测试=4：1

    return train_iter, test_iter  # 含有部分特征表示


def load_ind_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, train=False)
    return data_iter


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim=32, v_dim=32, num_heads=8):     # 32，32，4
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 64
        self.batch_size = 64
        self.emb_dim = 128

        self.embedding_seq = nn.Embedding(24, self.emb_dim, padding_idx=0)
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=1)
        self.gru_seq = nn.GRU(4260, self.hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)   #3160，3860，1900，2200
        self.cross_attention = CrossAttention(in_dim1=128, in_dim2=128)

        self.conv_cross = nn.Sequential(
            nn.Conv1d(133, 64, kernel_size=3, stride=1, padding=0),  # 64\32\16(D3)
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=0, dilation=1, return_indices=False, ceil_mode=False),
        )

        self.block1 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(2688,512),    # 2688\672,512\256
                                    nn.BatchNorm1d(512),
                                    nn.Dropout(0.6),
                                    nn.LeakyReLU(),
                                    nn.Linear(512, 128),
                                    )

        self.block2 = nn.Sequential(nn.Linear(128, 64),
                                    nn.BatchNorm1d(64),
                                    nn.Dropout(0.6),
                                    nn.ReLU(),
                                    nn.Linear(64, 2),
                                    nn.Softmax(dim=1))

    def forward(self, x, pos_embed, HF):
        output1 = self.embedding_seq(x) + pos_embed
        output1 = self.transformer_encoder_seq(output1)  # .permute(1, 0, 2)

        output2 = torch.unsqueeze(HF, dim=1)
        output2, hn = self.gru_seq(output2)

        output = self.cross_attention(output1, output2)
        output = self.conv_cross(output)

        output = self.block1(output)
        out = self.block2(output)

        return out,output


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    for x, pos, hf, y in data_iter:
        outputs,_  = net(x, pos, hf)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y.tolist()
        #rep_list.extend(rep.detach().numpy())
    performance, FPR, TPR, recall, precision = caculate_metric(pred_prob, label_pred, label_real)
    return performance, FPR, TPR, recall, precision


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    # F1
    if tp + fp == 0:
        Precision = 0
        F1 = 0
    else:
        Precision = float(tp) / (tp + fp)
        F1 = 2 * Precision * Recall / (Precision + Recall)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC, F1, AP]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, FPR, TPR, recall, precision


def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    l2_lambda = 0.0
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)

    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss



def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model



