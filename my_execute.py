import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import sys
from models import DGI, LogReg
from utils import process

dataset = 'philadelphia'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = float(sys.argv[1])
drop_prob = float(sys.argv[2])
hid_units = 128
sparse = True
nonlinearity = 'prelu'  # special name to separate parameters

# adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

adj, features, idx_train = process.my_load_data(dataset)
features, _ = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])

idx_train = torch.LongTensor(idx_train)

model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()

    idx_train = idx_train.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'best_dgi%s_%s_%s.pkl' % (dataset, l2_coef, drop_prob))
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi%s_%s_%s.pkl' % (dataset, l2_coef, drop_prob)))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
embeds = embeds.to("cpu", torch.double).numpy()
np.save("%s_%s_%s.emb" % (dataset, l2_coef, drop_prob), embeds)
