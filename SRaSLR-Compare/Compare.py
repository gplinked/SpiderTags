import time

import dgl
import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import networkx as nx
from scipy import sparse as sp
import torchmetrics
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertModel
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import gc

def getGraphData(G1,node_embeddings):
    edge_index = []
    edge_attr = []
    for edge in G1.edges():
        edge_index.append([int(edge[0]), int(edge[1])])
        edge_attr.append(G1[edge[0]][edge[1]]['weight'])
    # Convert edges and weights to Tensor tensors
    edge_index = torch.tensor(edge_index).T.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    edge_attr = torch.tensor(edge_attr).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Transform node embedding into Tensor tensor
    embedding_array = node_embeddings.vectors
    # Convert NumPy array to torch. sensor
    x = torch.tensor(embedding_array)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

class SRaSLRMMAADataset(torch.utils.data.Dataset):

    def __init__(self, input_ids, tokentype_ids, attention_mask, targets, node_ids, node_type, transform=None):
        self.input_ids = np.array(input_ids, dtype=np.int32)
        self.tokentype_ids = np.array(tokentype_ids, dtype=np.int32)
        self.attention_mask = np.array(attention_mask, dtype=np.int32)
        self.targets = np.array(targets, dtype=np.int32)
        self.node_ids = np.array(node_ids, dtype=np.int32)
        self.node_types = np.array(node_type, dtype=np.int32)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_id, tokentype_id, attention_mask, target = \
            self.input_ids[index], self.tokentype_ids[index], self.attention_mask[index], self.targets[index]
        node_id = self.node_ids[index]
        node_type = self.node_types[index]
        return torch.as_tensor(input_id).type(torch.LongTensor), torch.as_tensor(tokentype_id).type(torch.LongTensor), \
               torch.as_tensor(attention_mask).type(torch.LongTensor), \
               torch.as_tensor(target).type(torch.LongTensor), torch.as_tensor(node_id).type(
            torch.LongTensor), torch.as_tensor(node_type).type(torch.LongTensor)



class GCN(pl.LightningModule):
    def __init__(self, alpha, eps, matrix, graph, bert_checkpoint, in_feats, num_classes, node_embeddings, dense_dropout=0.5):
        super(GCN, self).__init__()

        #bert
        self.config = BertConfig.from_pretrained(bert_checkpoint)
        self.bert_encoder = BertModel.from_pretrained(bert_checkpoint, config=self.config)

        self.conv1 = GCNConv(in_feats, 256)
        self.layer_norm1 = nn.LayerNorm(256)
        self.conv2 = GCNConv(256, 128)
        self.layer_norm2 = nn.LayerNorm(128)
        self.conv3 = GCNConv(128, 50)
        self.batch_norm = nn.BatchNorm1d(50)
        self.graph = graph.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=50+self.config.hidden_size+1, out_features=50+self.config.hidden_size+1),
            nn.Dropout(dense_dropout),
            nn.Linear(in_features=50+self.config.hidden_size+1, out_features=num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.recall = torchmetrics.Recall(num_classes=num_classes, average='macro')
        self.preci = torchmetrics.Precision(num_classes=num_classes, average='macro')
        self.f1_score = torchmetrics.F1(num_classes=num_classes, average='macro')

        self.matrix = matrix
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=True)
        self.startTime = time.time()
        self.duration = 0
        self.node_embeddings = node_embeddings


    def gdc(self, A1):

        gdc_alpha = 1 + 9 * self.alpha  # 将alpha转换到[1,10]之间

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

    def forward(self, graph1, input_ids , token_type_ids, attention_mask, node_ids, node_type):
        bert_output = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_output = bert_output[0][:,0]

        #cat
        node_type = node_type.unsqueeze(1)
        data = self.graph.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        data.edge_attr = graph1.edge_attr
        data.edge_index = graph1.edge_index

        nodes = self.conv1(data.x, data.edge_index, data.edge_attr)
        nodes = self.layer_norm1(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv2(nodes, data.edge_index, data.edge_attr)
        nodes = self.layer_norm2(nodes.to(torch.float32))
        nodes = self.relu(nodes.to(torch.float))
        nodes = self.conv3(nodes, data.edge_index, data.edge_attr)
        nodes = self.batch_norm(nodes.to(torch.float32))

        nodes = nodes[node_ids]

        input = torch.cat((bert_output, nodes, node_type), 1)

        # Decoder forward
        h = self.decoder(input)

        return h

    def training_step(self, train_batch, batch_idx):
        # Obtain batch data
        input_ids, token_type_ids, attention_mask, targets, node_ids, node_type = train_batch

        # Input data into the model for forward calculation
        outputs = self.forward(self.graph, input_ids, token_type_ids, attention_mask, node_ids, node_type)

        # Calculate losses
        loss = self.loss_fn(outputs, targets)
        loss = loss.mean()

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_ids, token_type_ids, attention_mask, targets, node_ids, node_type = val_batch

        with torch.no_grad():
            outputs = self.forward(self.graph, input_ids, token_type_ids, attention_mask, node_ids, node_type)

        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        endTime = time.time()

        input_ids, token_type_ids, attention_mask, targets, node_ids, node_type = val_batch

        outputs = self.forward(self.graph, input_ids, token_type_ids, attention_mask, node_ids, node_type)

        output = outputs.to('cpu').numpy()
        target = targets.to('cpu').numpy()
        correct_top1, incorrect_top1 = 0, 0
        correct_top5, incorrect_top5 = 0, 0
        for o, t in zip(output, target):
            sorted_args = np.argsort(-o)
            if sorted_args[0] == t:
                correct_top1 += 1
            else:
                incorrect_top1 += 1
            if t in sorted_args[:5]:
                correct_top5 += 1
            else:
                incorrect_top5 += 1

        target = torch.from_numpy(target)
        loss = self.loss_fn(outputs, targets)
        loss = loss.mean()
        recall = self.recall(outputs, targets)
        preci = self.preci(outputs, targets)
        f1 = self.f1_score(outputs, targets)
        acc = self.accuracy(outputs, targets)

        print(f' Test loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Recall: {recall:.4f}, '
              f'Test Precision: {preci:.4f}, Test F1-score: {f1:.4f}')
        print(f'alpha: {self.alpha:.2f}')
        print(f'eps: {self.eps:.2f}')
        print("\n")

        with open("single-1", "a") as file:
            file.write(f' Test loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Recall: {recall:.4f}, '
                  f'Test Precision: {preci:.4f}, Test F1-score: {f1:.4f}')
            file.write(f'alpha: {self.alpha:.2f}')
            file.write(f'eps: {self.eps:.2f}')
            file.write("\n")

        # Record the value of the indicator
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)
        self.log('test_preci', preci, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
        self.duration = endTime - self.startTime
        self.log('duration', endTime - self.startTime, on_step=False, on_epoch=True)

        return {"correct_top1": correct_top1, "correct_top5": correct_top5, "incorrect_top1": incorrect_top1,
                "incorrect_top5": incorrect_top5, 'test_loss' : loss , 'test_acc': acc , 'test_recall': recall,
                'test_preci': preci, 'test_f1': f1 , 'duration' : endTime-self.startTime}

    def test_epoch_end(self, outputs):
        correct_top1, incorrect_top1 = 0, 0
        correct_top5, incorrect_top5 = 0, 0
        for out in outputs:
            correct_top1 += out["correct_top1"]
            incorrect_top1 += out["incorrect_top1"]
            correct_top5 += out["correct_top5"]
            incorrect_top5 += out["incorrect_top5"]
        print({"acc_top1": correct_top1 / (correct_top1 + incorrect_top1),
               "acc_top5": correct_top5 / (correct_top5 + incorrect_top5)})

        with open("single-1", "a") as file:
            file.write("acc_top1"+ str(correct_top1 / (correct_top1 + incorrect_top1)))
            file.write("acc_top5"+ str(correct_top5 / (correct_top5 + incorrect_top5)))

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        # Summarize ACC, Recall, Precision, and F-measure values for all test batches
        acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        preci = torch.stack([x['test_preci'] for x in outputs]).mean()
        f1 = torch.stack([x['test_f1'] for x in outputs]).mean()


        # Output the value of test indicators
        self.log('test_loss', avg_loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True)
        self.log('test_precision', preci, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)

        print(f' Test loss: {avg_loss:.4f}, Test Acc: {acc:.4f}, Test Recall: {recall:.4f}, '
              f'Test Precision: {preci:.4f}, Test F1-score: {f1:.4f}')
        print(f'Time elapsed: {self.duration:.2f} seconds')
        print(f'alpha: {self.alpha:.2f}')
        print(f'eps: {self.eps:.2f}')
        print("\n")

        with open("single-1", "a") as file:
            file.write(f' Test loss: {avg_loss:.4f}, Test Acc: {acc:.4f}, Test Recall: {recall:.4f}, '
                  f'Test Precision: {preci:.4f}, Test F1-score: {f1:.4f}')
            file.write(f'Time elapsed: {self.duration:.2f} seconds')
            file.write(f'alpha: {self.alpha:.2f}')
            file.write(f'eps: {self.eps:.2f}')
            file.write("\n")


    def on_epoch_end(self):
        adj_matrix_diffused1 = self.gdc(self.matrix)
        array_A1 = adj_matrix_diffused1.cpu().detach().numpy()
        scr_A1 = sp.csr_matrix(array_A1)
        g1 = nx.from_scipy_sparse_matrix(scr_A1)
        self.graph = getGraphData(g1, self.node_embeddings)
        print('edges:' ,g1.number_of_edges(),"\n")
        print(f'alpha: {self.alpha:.2f}')
        #print(f'alpha.grad: {self.alpha.retain_grad():.2f}')
        print(f'eps: {self.eps:.2f}')
        with open("single-1", "a") as file:
            file.write('edges:'+ str(g1.number_of_edges()) + "\n")
            file.write(f'alpha: {self.alpha:.2f}')
            file.write(f'eps: {self.eps:.2f}')

    def configure_optimizers(self):
        params = [
            {'params': self.parameters(), 'lr': 3e-5},
            {'params': self.alpha, 'lr': 0.01},
            {'params': self.eps, 'lr': 0.01}
        ]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return self.optimizer

# Define hyperparameters
NUM_CLASSES = 255
EPOCHS = 50
A_ALPHA = 0.40
A_EPS = 0.01
P = 1.5
Q = 0.5

#！
model_path = './bert_model/'
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load graph and create DGL graph
G1 = nx.read_weighted_edgelist('./data/N-mmaa_frequency.edgelist')
PATH = 'data/embed/mm_aa_freq_100.txt'
node_embeddings = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False)
#Obtaining Graph Data
graph = getGraphData(G1,node_embeddings)
#Obtain adjacency matrix to graph diffusion
A1 = nx.to_numpy_matrix(G1)
np_array1 = np.asarray(A1)
adj_matrix1 = nx.to_scipy_sparse_matrix(G1, format="csr")

#Data
X = list(range(4099))
# train_ids, test_ids = train_test_split(X, test_size=0.1, random_state=1)
# train_ids, val_ids = train_test_split(train_ids, test_size=0.5, random_state=1)
train_ids, test_ids = train_test_split(X, test_size=0.2, random_state=1)
train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=1)
train_nodes = [node_embeddings.vocab[str(n)].index for n in train_ids]
val_nodes = [node_embeddings.vocab[str(n)].index for n in val_ids]
test_nodes = [node_embeddings.vocab[str(n)].index for n in test_ids]

titles = [line.strip().split(' =->= ')[1] for line in open('data/titles_encoded.txt', 'r').readlines()]
titles = [0 if line.startswith('Mashup:') else 1 for line in titles]
node_types = np.array(titles)

text_all = np.array([line.strip().split(' =->= ')[1] for line in open('data/dscps_encoded.txt', 'r').readlines()])
tags_all = np.array([int(line.strip()) for line in open('data/tags_id.txt').readlines()])


MAX_LENGTH = 300
train_X = tokenizer(list(text_all[train_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
train_Y = tags_all[train_ids]
train_dataset = SRaSLRMMAADataset(train_X['input_ids'], train_X['token_type_ids'], train_X['attention_mask'], train_Y,
                                  train_nodes, node_types[train_ids])

val_X = tokenizer(list(text_all[val_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
val_Y = tags_all[val_ids]
val_dataset = SRaSLRMMAADataset(val_X['input_ids'], val_X['token_type_ids'], val_X['attention_mask'], val_Y, val_nodes,
                                node_types[val_ids])

test_X = tokenizer(list(text_all[test_ids]), padding=True, truncation=True, max_length=MAX_LENGTH)
test_Y = tags_all[test_ids]
test_dataset = SRaSLRMMAADataset(test_X['input_ids'], test_X['token_type_ids'], test_X['attention_mask'], test_Y,
                                 test_nodes, node_types[test_ids])

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=4)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=4)

model = GCN(A_ALPHA, A_EPS, np_array1, graph, model_path, 100, NUM_CLASSES, node_embeddings)  # main.GCN?修改

trainer = pl.Trainer(max_epochs=15, gpus='1')
# trainer = pl.Trainer(max_epochs=9, gpus='0')
trainer.fit(model, train_dl, val_dl)
trainer.test(model, test_dl)
