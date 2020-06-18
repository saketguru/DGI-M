import os
from time import gmtime, strftime

parent_dir = "../link_prediction_data/"
# datasets = ["wisconsin_80_20", "blogcatalog_80_20", "flickr_80_20", "p2p-gnutella31_80_20", "texas_80_20",
#            "wikivote_80_20", "youtube_80_20", "co-author_80_20", "microsoft_80_20", "ppi_80_20", "washington_80_20",
#            "cornell_80_20", "pubmed_80_20", "wikipedia_80_20"]

embedding_par = "../link_prediction_embeddings/"
datasets = ["wisconsin", "blogcatalog", "flickr", "p2p-gnutella31", "texas",
            "wikivote", "youtube", "co-author", "microsoft", "ppi", "washington",
            "cornell", "pubmed", "wikipedia"]

input_graph_dirs = []
data_names = []
output_file_names = []
input_embedding0_dirs = []
embedding0_file_names = []

for dataset in datasets:
    network = "../datasets/%s.mat" % dataset
    embedding0_file_name = embedding_par + dataset + ".npy"

    os.system("qsub -v emb=%s,nw=%s,db=%s eval_nc_embeddings.pbs" % (embedding0_file_name, network, dataset))
    

