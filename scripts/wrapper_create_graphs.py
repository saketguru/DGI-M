import os

# Link prediction
# parent_dir = "../datasets/link_prediction_data/"
# folders = ["wisconsin_80_20", "blogcatalog_80_20", "flickr_80_20", "p2p-gnutella31_80_20", "texas_80_20",
#            "wikivote_80_20", "youtube_80_20", "co-author_80_20", "microsoft_80_20", "ppi_80_20", "washington_80_20",
#            "cornell_80_20", "pubmed_80_20", "wikipedia_80_20"]
#
# for folder in folders:
#     mid_parent_dir = parent_dir + folder + "/"
#     for i in range(0, 5):
#         fname = mid_parent_dir + "fold_%s.edgelist" % i
#         if not os.path.exists(fname):
#             print (fname)
#
#         write_name = "../data/" + folder + "_fold_%s.graph" % i
#         os.system("qsub -v fname=%s,wfname=%s create_graph.pbs" % (fname, write_name))

# Node classification

parent_dir = "../datasets/"
datasets = ["wisconsin", "blogcatalog", "flickr", "p2p-gnutella31", "texas",
            "wikivote", "youtube", "co-author", "microsoft", "ppi", "washington",
            "cornell", "pubmed", "wikipedia"]

for dataset in datasets:
    fname = parent_dir + dataset + ".edgelist"

    if not os.path.exists(fname):
        continue

    write_name = "../data/" + dataset + ".graph"

    os.system("qsub -v fname=%s,wfname=%s create_graph.pbs" % (fname, write_name))
