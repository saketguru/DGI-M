import os
from time import gmtime, strftime

parent_dir = "../link_prediction_data/"
# datasets = ["wisconsin_80_20", "blogcatalog_80_20", "flickr_80_20", "p2p-gnutella31_80_20", "texas_80_20",
#            "wikivote_80_20", "youtube_80_20", "co-author_80_20", "microsoft_80_20", "ppi_80_20", "washington_80_20",
#            "cornell_80_20", "pubmed_80_20", "wikipedia_80_20"]

embedding_par = "../link_prediction_embeddings/"
datasets = ["co-author_80_20", "cornell_80_20", "flickr_80_20", "microsoft_80_20",
            "p2p-gnutella31_80_20",
            "pubmed_80_20", "washington_80_20", "wisconsin_80_20", "youtube_80_20"]
# "texas_80_20"]
input_graph_dirs = []
data_names = []
output_file_names = []
input_embedding0_dirs = []
embedding0_file_names = []

for dataset in datasets:
    input_graph_dir = parent_dir + dataset
    data_name = dataset
    output_file_name = dataset + "_result.csv"
    input_embedding0_dir = embedding_par
    embedding0_file_name = dataset + "_fold"

    os.system("qsub -v igd=%s,data=%s,ofn=%s,ied0=%s,iefn=%s eval_embeddings.pbs" % (
        input_graph_dir, data_name, output_file_name, input_embedding0_dir, embedding0_file_name))
