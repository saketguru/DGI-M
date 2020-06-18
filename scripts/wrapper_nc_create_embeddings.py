import os
from time import gmtime, strftime

parent_dir = "../data/"
datasets = ["wisconsin", "blogcatalog", "flickr", "p2p-gnutella31", "texas",
            "wikivote", "youtube", "co-author", "microsoft", "ppi", "washington",
            "cornell", "pubmed", "wikipedia"]

for dataset in datasets:
    graph_name = dataset
    os.system("qsub -v db=%s,t=%s job_osc.pbs" % (graph_name, "_" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())))
