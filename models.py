import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data
import os

device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"

# Siamese LSTM
class LSTM(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dims=3, num_classes=5, num_layers=1, dropout=0.2, bidirectional=False,use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" \
                         f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" \
                         f"_dropout={dropout}"

        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_dims

        self.num_classes = num_classes
        self.use_layernorm = use_layernorm

        self.d_model = num_layers * hidden_dims

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_layers)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class = nn.Linear(hidden_dims * num_layers, num_classes, bias=True)

        # below layers needed since 2603 and 2803 are trained with these layers altough not using them
        self.hidden1 = nn.Linear(hidden_dims * num_layers * 2, (hidden_dims // 2) * num_layers, bias=True)
        self.hidden2 = nn.Linear((hidden_dims // 2) * num_layers, 2, bias=True)
        self.softmax = nn.Softmax(0)

    def logits_one(self,x):
        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm.forward(x)

        h, c = last_state_list

        nlayers, batchsize, n_hidden = c.shape
        h = self.clayernorm(c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden))

        return h

    def forward(self, x1, x2):
        out1 = self.logits_one(x1)
        out2 = self.logits_one(x2)

        return out1, out2

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

# ANN fed by Siamese LSTM
class ClassificationModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(ClassificationModel,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self.input_dim,self.hidden_dim,bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim,self.output_dim,bias=True)

    def forward(self,X):
        out = self.linear1(X)
        out = self.relu(out)
        out = self.linear2(out)

        return out

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot