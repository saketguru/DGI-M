#!/usr/bin/env bash

l2_coefs=(0.0 0.1 0.2)
drop_probs=(0.0 0.25 0.5)




for l2 in ${l2_coefs[@]}
do
for dp in ${drop_probs[@]}
do

sbatch ri2.sh $l2 $dp
#exit

#python eval_nationwide.py "philadelphia_"$l2"_"$dp".emb.npy"
#exit
done
done
