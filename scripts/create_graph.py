import sys
sys.path.append("..")
import pickle as pkl

import mutils

    # mutils import _read_graph_from_edgelist, graph2nx
import networkx as nx
import sys


class Ctrl:
    pass


ctrl = Ctrl()
ctrl.debug_mode = False

fname = sys.argv[1]
wfname = sys.argv[2]

graph, mapping = mutils._read_graph_from_edgelist(ctrl, fname)
print (fname)
print (graph.node_num)

G = mutils.graph2nx(graph)
pkl.dump(nx.to_scipy_sparse_matrix(G), open(wfname, "wb"))

# nx.write_edgelist(G, "columbus_graph.csv", delimiter=",")
# gg  = dict()
# for i, nodes in enumerate(nx.generate_adjlist(G)):
#     gg[i]=nodes
# print(i)
# A = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))
