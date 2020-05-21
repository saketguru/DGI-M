import pickle as pkl
from mutils import _read_graph_from_edgelist, graph2nx
import networkx as nx


class Ctrl:
    pass


ctrl = Ctrl()
ctrl.debug_mode = False

graph, mapping = _read_graph_from_edgelist(ctrl, "philadelphia_graph.csv") #""train_nominatim_graph_0.txt")

G = graph2nx(graph)


# nx.write_edgelist(G, "columbus_graph.csv", delimiter=",")
#
# gg  = dict()
# for i, nodes in enumerate(nx.generate_adjlist(G)):
#     gg[i]=nodes
# print(i)
# A = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))


pkl.dump(nx.to_scipy_sparse_matrix(G), open("data/philadelphia.graph", "wb"))
