import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool as gap
from torch.nn import LayerNorm, Parameter
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict
import argparse
from util import *

parser = argparse.ArgumentParser(description='Training Script')  
parser.add_argument('--adj1', type=float, default=0.2)
parser.add_argument('--adj2', type=float, default=0.2)
parser.add_argument('--adj3', type=float, default=0.2)
parser.add_argument('--adj_AHBA', type=float, default=0.3)
parser.add_argument('--batch_size_', type=int, default=24)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default=0.6) 
args = parser.parse_args()


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class Fusion(nn.Module):
    def __init__(self, num_class, num_views, hidden_dim, dropout, in_dim):
        super().__init__()
        self.gat1 =  GAT(dropout=args.dropout, alpha=args.alpha, dim=116)
        self.gat_1 = GAT(dropout=args.dropout, alpha=args.alpha, dim=116)
        self.gat2 =  GAT(dropout=args.dropout, alpha=args.alpha, dim=116)
        self.gat_2 = GAT(dropout=args.dropout, alpha=args.alpha, dim=116)
        self.gat3 =  GAT(dropout=args.dropout, alpha=args.alpha, dim=116)
        self.gat_3 = GAT(dropout=args.dropout, alpha=args.alpha, dim=116)
        self.CrossAtt = CrossModalAttention(64, 64, 8)
        self.lin_2 =nn.Sequential(nn.Linear(64, 32),nn.ReLU(),nn.Linear(32, 2))
        self.lin_64= LinearLayer(64, 64)
        self.dro = nn.Dropout(p=0.25)
        self.attention = SelfAttention(input_dim=64)
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.TCPConfidenceLayer = nn.ModuleList([TCPConfidenceLayer(hidden_dim[0]) for _ in range(self.views)])
        self.Fan_TCPClassifierLayer = nn.ModuleList([Fan_TCPClassifierLayer(hidden_dim[0]) for _ in range(self.views)])

        self.TCPClassifierLayer = nn.ModuleList([TCPClassifierLayer(hidden_dim[0]) for _ in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1],num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, omic1, omic2, omic3, adj1, adj2, adj3, adj_AHBA, label=None, infer=False):
        output_1, gat_output_1 = self.gat_1(omic1, adj_AHBA)
        output_2, gat_output_2 = self.gat_2(omic2, adj_AHBA)
        output_3, gat_output_3 = self.gat_3(omic3, adj_AHBA)
        feature = dict() 
        feature[0] = output_1
        feature[1] = output_2
        feature[2] = output_3
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_function = nn.CrossEntropyLoss()
        FeatureInfo, TCPLogit, TCPConfidence, Fan_TCPConfidence,Pingjun_TCP= dict(), dict(), dict(),dict(), dict()
        p_target_dict = dict()
        for view in range(self.views):
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            Fan_TCPConfidence[view] = self.Fan_TCPClassifierLayer[view](feature[view])
            Pingjun_TCP[view] = 2/(1/TCPConfidence[view]+1/(1-Fan_TCPConfidence[view]))
            feature[view] = feature[view] * Pingjun_TCP[view]
        feature_TCP = dict()
        feature_TCP[0] = self.CrossAtt(feature[0],feature[1]).squeeze(1)
        feature_TCP[1] = self.CrossAtt(feature[1],feature[2]).squeeze(1)
        feature_TCP[2] = self.CrossAtt(feature[0],feature[2]).squeeze(1)

        MMfeature = torch.cat([i for i in feature_TCP.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        loss_gat1 = loss_function(gat_output_1,label)
        loss_gat2 = loss_function(gat_output_2,label)
        loss_gat3 = loss_function(gat_output_3,label)
        gat_loss = dict()
        gat_loss[0], gat_loss[1], gat_loss[2] = loss_gat1, loss_gat2, loss_gat3
        for view in range(self.views):
            MMLoss = MMLoss + gat_loss[view]
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            Fan_p_target= 1 - p_target

            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            Fan_confidence_loss =  torch.mean(F.mse_loss(Fan_TCPConfidence[view].view(-1), Fan_p_target))

            MMLoss = MMLoss + confidence_loss + Fan_confidence_loss
        MM_prob = F.softmax(MMlogit, dim=1)
        MM_tcp = MM_prob.max(dim=1)[0] 
        return MMLoss, MMlogit, MM_tcp, gat_output_1, gat_output_2, gat_output_3, output_1, output_2, output_3

    def infer(self, omic1, omic2, omic3, adj1, adj2, adj3, adj_AHBA):
        MMlogit = self.forward(omic1, omic2, omic3, adj1, adj2, adj3, adj_AHBA, infer=True)
        return MMlogit

class TCPConfidenceLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TCPConfidenceLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 356) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(356, 128)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(128, 1)  

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x  
class Fan_TCPClassifierLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(Fan_TCPClassifierLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 356) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(356, 128) 
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 1)  

        self.sigmoid = nn.Sigmoid() 

        self.softplus = nn.Softplus() 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x  
class TCPClassifierLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TCPClassifierLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 356) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(356, 128) 
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 2) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x  
class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([input_dim])).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output
class CrossModalAttention(nn.Module):
    def __init__(self, in_dim,d_model, num_heads):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W1_q = nn.Linear(in_dim, d_model)
        self.W1_k = nn.Linear(in_dim, d_model)
        self.W1_v = nn.Linear(in_dim, d_model)
        self.W2_q = nn.Linear(in_dim, d_model)
        self.W2_k = nn.Linear(in_dim, d_model)
        self.W2_v = nn.Linear(in_dim, d_model)

        self.fc = nn.Linear(2 * self.num_heads * self.d_v, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        dk = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        Q1 = self.W1_q(x1).view(batch_size, -1, self.num_heads, self.d_k)
        K1 = self.W1_k(x1).view(batch_size, -1, self.num_heads, self.d_k)
        V1 = self.W1_v(x1).view(batch_size, -1, self.num_heads, self.d_v)
        Q2 = self.W2_q(x2).view(batch_size, -1, self.num_heads, self.d_k)
        K2 = self.W2_k(x2).view(batch_size, -1, self.num_heads, self.d_k)
        V2 = self.W2_v(x2).view(batch_size, -1, self.num_heads, self.d_v)
        Q1 = Q1.transpose(1, 2)
        K1 = K1.transpose(1, 2)
        V1 = V1.transpose(1, 2)
        Q2 = Q2.transpose(1, 2)
        K2 = K2.transpose(1, 2)
        V2 = V2.transpose(1, 2)
        output1, _ = self.scaled_dot_product_attention(Q1, K2, V2)
        output2, _ = self.scaled_dot_product_attention(Q2, K1, V1)
        output = torch.cat([output1, output2], dim=-1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, 2 * self.num_heads * self.d_v)
        output = self.fc(output)

        return output
class GAT(nn.Module):
    def __init__(self, dropout, alpha, dim):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.act = define_act_layer(act_type='none')
        self.dim = dim
        self.nhids = [8, 16, 12]
        self.nheads = [4, 3, 4]
        self.fc_dim = [348,256, 64, 32]
        self.CrossAtt = CrossModalAttention(116, 116, 116)
        self.attentions1 = [GraphAttentionLayer(
            1, self.nhids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[0])]
        for i, attention1 in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention1)

        self.attentions2 = [GraphAttentionLayer(
            self.nhids[0] * self.nheads[0], self.nhids[1], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[1])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)

        self.attentions3 = [GraphAttentionLayer(
            self.nhids[1] * self.nheads[1], self.nhids[2], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[2])]
        for i, attention3 in enumerate(self.attentions3):
            self.add_module('attention3_{}'.format(i), attention3)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.pool1 = torch.nn.Linear(self.nhids[0] * self.nheads[0], 1)
        self.pool2 = torch.nn.Linear(self.nhids[1] * self.nheads[1], 1)
        self.pool3 = torch.nn.Linear(self.nhids[2] * self.nheads[2], 1)

        lin_input_dim = 3 * self.dim
        self.fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc1.apply(xavier_init)

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc2.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], self.fc_dim[3]),
            nn.Mish(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_dim[3], 2))
        self.fc5.apply(xavier_init)

    def forward(self, x, adj):


        x0 = torch.mean(x, dim=-1)

        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)

        x1 = self.pool1(x).squeeze(-1)


        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)

        x2 = self.pool2(x).squeeze(-1)

        x = torch.cat([x0, x1, x2], dim=1)

        x = self.fc1(x) 
        x = self.fc2(x) 
        x1 = self.fc3(x) 
        x = self.fc4(x1) 
        x = self.fc5(x) 

        output = x1  
        gat_output = x 

        return output, gat_output
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        adj = adj.to(h.device) 
        """
        :param h: (batch_zize, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_zize, number_nodes, out_features)
        """
        # batchwise matrix multiplication
        Wh = torch.matmul(h, self.W)  # (batch_zize, number_nodes, in_features) * (in_features, out_features) -> (batch_zize, number_nodes, out_features)
        e = self.prepare_batch(Wh)  # (batch_zize, number_nodes, number_nodes)

        # (batch_zize, number_nodes, number_nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        # (batch_zize, number_nodes, number_nodes)
        attention = torch.where(adj > 0, e, zero_vec)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.softmax(attention, dim=-1)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # batched matrix multiplication (batch_zize, number_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        B, N, E = Wh.shape  # (B, N, N)

        # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

        # broadcast add (B, N, 1) + (B, 1, N)
        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
