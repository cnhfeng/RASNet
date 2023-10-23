import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution
import math
from torch.nn.parameter import Parameter


"""
Our model
"""

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class Aggregation(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(Aggregation, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU()
        )
        
        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs):
        gates = self.gate_layer(self.h1(seqs))
        output = F.sigmoid(gates)

        return output

class AllRecDrugModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        fai,
        emb_dim=64,
        device=torch.device("cuda:0"),
    ):
        super(AllRecDrugModel, self).__init__()
        self.device = device
        self.fai = fai
        self.emb = emb_dim
        self.voc_size = vocab_size
        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(3)]
        )
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )
        
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        self.poly = Aggregation(emb_dim * 2)

        self.classication = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 2, vocab_size[2]))

        self.layernorm = nn.LayerNorm(emb_dim)
        # self.layernorm1 = nn.LayerNorm(emb_dim * 2)

        # med pr
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter1 = Parameter(torch.FloatTensor(1), requires_grad=True)

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        # self.init_weights()


    def forward(self,input):
        i1_seq = []
        i2_seq = []
        preser = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        
        for adm in input:
            i1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            i2 = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        # patient_representations = torch.cat([i1_seq[:,-1:,:], i2_seq[:,-1:,:]], dim=-1).squeeze(dim=0) # (1,dim*2)
        # cur_query = self.query1(patient_representations)
        # change_storage = torch.zeros((1,self.emb)).to(device=self.device)

        if len(input) >= 2:
            #当前健康嵌入
            patient_representation = torch.concatenate([i1_seq, i2_seq], dim=-1).squeeze(dim=0)

            cur_query = patient_representation[-1:,:]

            # 获取与当前患者相似的第i次诊断
            poly_cur = self.poly(cur_query)
            for i in range(len(input)-1):

                poly_his = self.poly(patient_representation[i])
                s = abs(poly_cur - poly_his)
                if s <= self.fai:
                    preser.append(i)

            if not preser:
                i1_seq = i1_seq[:,-1:,:]
                i2_seq = i2_seq[:,-1:,:]

            else:
                # residuals_drug_emb = torch.cat([residuals_drug_emb[i] for i in preser],dim=0)
                preser.append(len(input)-1)
                i1_seq = torch.cat([i1_seq[:,i:i+1,:] for i in preser], dim=1)
                i2_seq = torch.cat([i2_seq[:,i:i+1,:] for i in preser], dim=1)

        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)

        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*2)

        query = self.query(patient_representations)[-1:,:]

        safe_gcn = self.ehr_gcn() - self.inter1 * self.ddi_gcn()

        med_base = self.embeddings[2](torch.LongTensor([x for x in range(self.voc_size[2])]).to(self.device))

        drug_gcn = torch.Tensor(safe_gcn + med_base).to(device=self.device) 

        key_weights1 = torch.softmax(torch.mm(query , drug_gcn.t()), dim=-1)  # (1, size)
        med_result = torch.mm(key_weights1, drug_gcn)  # (1, dim)

        final_representations = torch.cat([self.layernorm(query), med_result], dim=-1)
        result = self.classication(final_representations)
        
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    # def init_weights(self):
    #     """Initialize weights."""
    #     initrange = 0.1
    #     for item in self.embeddings:
    #         item.weight.data.uniform_(-initrange, initrange)

