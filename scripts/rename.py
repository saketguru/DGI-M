import os

files = os.listdir("../link_prediction_embeddings/")
for file in files:
    if not ".pkl.emb." in file:
        continue
    os.rename("../link_prediction_embeddings/" + file,
              "../link_prediction_embeddings/" + file.replace(".pkl.emb.", "."))
