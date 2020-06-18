import os
from time import gmtime, strftime

parent_dir = "../datasets/link_prediction_data/"
# datasets = ["wisconsin_80_20", "blogcatalog_80_20", "flickr_80_20", "p2p-gnutella31_80_20", "texas_80_20",
#            "wikivote_80_20", "youtube_80_20", "co-author_80_20", "microsoft_80_20", "ppi_80_20", "washington_80_20",
#            "cornell_80_20", "pubmed_80_20", "wikipedia_80_20"]

datasets = ["p2p-gnutella31_80_20", "flickr_80_20"] #youtube_80_20
for dataset in datasets:
    for i in range(0, 5):
        graph_name = dataset + "_fold_%s" % i
        os.system("qsub -v db=%s,t=%s job_osc.pbs" % (graph_name, "_" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())))
        # exit()
