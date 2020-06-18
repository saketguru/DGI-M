#!/bin/sh
#SBATCH -p batch-storage
#SBATCH -t 02-00:00:00
#SBATCH -N 1
#SBATCH -o slurms/slurm-%A.out # STDOUT
#SBATCH -e slurms/slurm-%A.err # STDERR


#-bdw-v100

conda activate gpstrails
#python my_execute.py $1 $2


#python eval_nationwide.py "atlanta_"$1"_"$2".emb.npy"
#python eval_nationwide.py "pittsburgh_"$1"_"$2".emb.npy"
python eval_nationwide.py "philadelphia_"$1"_"$2".emb.npy"

