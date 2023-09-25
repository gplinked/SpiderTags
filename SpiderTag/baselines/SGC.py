import json
import pickle
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Data.InputData import test_dataWord2Vec128, val_dataWord2Vec128, train_dataWord2Vec128
from torchmetrics import F1, Precision, Recall
from torch_geometric.nn import  global_mean_pool, SGConv
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
import networkx as nx
from scipy import sparse as sp
from torch_geometric.utils import to_networkx

def select_negative_samples(label, negative_sample_ratio: int = 4):
    num_candidate = label.size(0)
    positive_idx = label.nonzero(as_tuple=True)[0]
    positive_idx = positive_idx.cpu().numpy()
    size = negative_sample_ratio * len(positive_idx)
    if size>len(label)-len(positive_idx):
        size = len(label)-len(positive_idx)
    negative_idx = np.random.choice(np.delete(np.arange(num_candidate), positive_idx),
                                    size=size, replace=False)
    sample_idx = np.concatenate((positive_idx, negative_idx), axis=None)
    label_new = torch.tensor([1] * len(positive_idx) + [0] * len(negative_idx), dtype=torch.float32)
    return positive_idx, negative_idx, sample_idx, label_new.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

def getGraphData(G1,node_embeddings):
    edge_index = []
    edge_attr = []
    for edge in G1.edges():
        edge_index.append([int(edge[0]), int(edge[1])])
        edge_attr.append(G1[edge[0]][edge[1]]['weight'])

    edge_index = torch.tensor(edge_index).T.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #edge_weight=np.array(edge_attr.cpu())
    edge_attr = torch.tensor(edge_attr).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    embedding_array = node_embeddings

    x = torch.tensor(embedding_array)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

class FC(pl.LightningModule):

    def __init__(self, graph ,matrix, alpha , eps, k):
        super(FC, self).__init__()
        self.conv1 = SGConv(128, 75, K=k, cached=False)
        self.conv2 = SGConv(75, 50, K=k, cached=False)
        self.graph = graph.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.register_buffer('labels', torch.zeros(50))
        self.register_buffer('global_mean_pool_batch', torch.tensor([0]))
        self.relu = nn.ReLU()
        self.linear_label = nn.Sequential(
            nn.Linear(178, 89),
            nn.ReLU(),
            nn.Linear(89, 1),
            nn.Sigmoid(),
        )
        params = torch.full([2], 0.5, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.f1 = F1(threshold=0.5)
        self.pre = Precision(average='micro')
        self.recall = Recall(average='micro')


        self.accuracy = torchmetrics.Accuracy()
        self.startTime = None
        self.matrix = matrix
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=True)

    def forward(self, x, stage, negative_sample):

        adj_matrix_diffused1 = self.gdc(self.matrix)
        array_A1 = adj_matrix_diffused1.cpu().detach().numpy()
        scr_A1 = sp.csr_matrix(array_A1)
        g1 = nx.from_scipy_sparse_matrix(scr_A1)
        self.graph = getGraphData(g1, self.graph.x)
        data = self.graph
        nodes = self.conv1(data.x, data.edge_index, data.edge_weight)
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv2(nodes, data.edge_index, data.edge_weight)
        nodes = self.relu(nodes.to(torch.float))

        y1 = torch.zeros(10)
        n = np.arange(len(nodes))
        if stage == "train":
            n = negative_sample
            x = x.repeat(1, 1)

        for i in n:
            node = nodes[i].repeat(len(x), 1)
            text_node = torch.cat([x, node], dim=1).to(torch.float)
            answer = self.linear_label(text_node)
            if i == n[0]:
                y1 = answer
            else:
                y1 = torch.cat((y1, answer), dim=1)

        graph = global_mean_pool(nodes, batch=self.global_mean_pool_batch)
        graph = graph.repeat(len(x), 1)
        labels = self.labels
        labels = labels.repeat(len(x), 1)
        if stage == "train":
            y1 = y1.squeeze(0)
            for i in range(y1.size(0)):
                if y1[i] > 0.5:
                    labels[0][n[i]] = 1
        else:
            for i in range(len(y1)):
                for j in range(len(y1[0])):
                    if y1[i][j] > 0.5:
                        labels[i][j] = 1

        return y1

    def on_train_start(self) :
        self.startTime = time.time()

    def training_step(self, train_batch, batch_idx):
        sequences, label, newLabel, _ = train_batch

        loss1  = 0, 0
        num_sample = 0

        positive_idx_new_label, negative_idx_new_label, sample_idx_new_label, label_new_label = select_negative_samples(newLabel, 1)
        sequences_sample = sequences[sample_idx_new_label, :]
        new_label_num = 0

        for api, api_labels, api_new_label in zip(sequences, label, newLabel):
            positive_idx, negative_idx, sample_idx, label_new = select_negative_samples(api_labels)
            output1 = self(api, "train", sample_idx)
            num_sample += len(sample_idx)
            if len(label_new) == 0:
                loss1 += 0
            else:
                loss1 += F.binary_cross_entropy(output1.squeeze(0), label_new.to(torch.float32), size_average=False, reduction='sum')

        loss = 0.5/(self.params[0] ** 2)*(loss1/num_sample)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequences, label, newLabel, _ = val_batch
        output1 = self(sequences, "val", 0)

        label = label.to(torch.float32)
        loss1 = F.binary_cross_entropy(output1, label).mean()
        loss = 0.5/(self.params[0] ** 2)*loss1
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        endTime = time.time()
        sequences, labels, newLabel, _ = test_batch
        output1 = self(sequences, "test", 0)

        target1 = labels.to(torch.int)
        target2 = newLabel.to(torch.int)

        loss1 = F.binary_cross_entropy(output1,target1.float()).mean()

        f1_R = self.f1(output1, target1)
        p_R = self.pre(output1, target1)
        r_R = self.recall(output1, target1)

        acc1 = self.accuracy(output1, target1)

        self.log('loss',loss1, on_step=False, on_epoch=True)
        self.log('test_acc1', acc1, on_step=False, on_epoch=True)
        duration = endTime - self.startTime
        self.log('duration', endTime - self.startTime, on_step=False, on_epoch=True)

        return {'loss':loss1,
                'test_acc1':acc1,
                'duration': endTime - self.startTime
                }



    def test_epoch_end(self, outputs):
        i, F1_R,  Precision_R, Recall_R= 0, 0, 0, 0
        for out in outputs:
            i += 1
            F1_R += out["F1_R"]
            Precision_R += out["Precision_R"]
            Recall_R += out["Recall_R"]
        print()
        print({"F1_R": F1_R/i, "Precision_R": Precision_R/i, "Recall_R": Recall_R/i})

        loss1 = torch.stack([x['loss'] for x in outputs]).mean()

        acc1 = torch.stack([x['test_acc1'] for x in outputs]).mean()

        self.log('loss', loss1, on_step=False, on_epoch=True)
        self.log('test_acc1', acc1, on_step=False, on_epoch=True)

        self.log("F1_R", F1_R / i, on_step=False, on_epoch=True)
        self.log("Precision_R", Precision_R / i, on_step=False, on_epoch=True)
        self.log("Recall_R",  Recall_R / i , on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def on_epoch_end(self):
        adj_matrix_diffused1 = self.gdc(self.matrix)
        array_A1 = adj_matrix_diffused1.cpu().detach().numpy()
        scr_A1 = sp.csr_matrix(array_A1)
        g1 = nx.from_scipy_sparse_matrix(scr_A1)
        self.graph = getGraphData(g1, self.graph.x)
        print('edges:' ,g1.number_of_edges(),"\n")
        print(f'alpha: {self.alpha:.2f}')
        print(f'eps: {self.eps:.2f}')
        with open("single-1", "a") as file:
            file.write('edges:'+ str(g1.number_of_edges()) + "\n")
            file.write(f'alpha: {self.alpha:.2f}')
            file.write(f'eps: {self.eps:.2f}')

    def gdc(self, A1):

        gdc_alpha = 1 + 9 * self.alpha

        num_nodes = A1.shape[0]
        A_tilde1 = A1 + np.eye(num_nodes)
        D_tilde1 = np.diag(1 / np.sqrt(A_tilde1.sum(axis=1)))
        H1 = D_tilde1 @ A_tilde1 @ D_tilde1
        H1 = torch.from_numpy(H1).float().to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        I1 = torch.eye(num_nodes).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        matrix1 = gdc_alpha * torch.inverse(I1 - (1 - gdc_alpha) * H1)
        matrix1[matrix1 < self.eps] = 0.
        norm1 = matrix1.sum(axis=0)
        norm1[norm1 <= 0] = 1  # avoid dividing by zero

        return matrix1/norm1

seed_everything(5)

train_dataset = train_dataWord2Vec128()
val_dataset = val_dataWord2Vec128()
test_dataset = test_dataWord2Vec128()

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4)



#fn = '..//myData//final_data//graph.pkl'
fn = '..//myData//final_data//data.pkl'
with open(fn, 'rb+') as f:
    graph = pickle.load(f)
graph.x = torch.tensor(graph.x, dtype=torch.float32)
graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
graph.edge_attr = torch.tensor(graph.edge_attr)

A_ALPHA = 0.50
A_EPS = 0.05
G1 = to_networkx(graph)
A1 = nx.to_numpy_matrix(G1)
np_array1 = np.asarray(A1)
adj_matrix1 = nx.to_scipy_sparse_matrix(G1, format="csr")

#graph = processNPY(graph)
model = FC(graph, np_array1, A_ALPHA, A_EPS, 15)

trainer = pl.Trainer(max_epochs=20, gpus='1')
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
print("over")
