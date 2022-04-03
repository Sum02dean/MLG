#!/usr/bin/env bash
#SBATCH --cpus-per-task=8  #8
#SBATCH --mem=32G   #50G
#SBATCH --time=24:00:00  #48:00:00
#SBATCH --output /net/cephfs/shares/von-mering.imls.uzh/tao/alphafold_output/slurm_reports/output_%A_%a.txt
#SBATCH --error /net/cephfs/shares/von-mering.imls.uzh/tao/alphafold_output/slurm_reports/error_%A_%a.txt

# chmod +x 
#module load generic #this should be outside this for sciencecluster to decide which particiton to use 
# we neeed very large memory and many CPUs so 
module load HPC



source activate /data/tfang/conda-envs/py309_MLG

cd /home/cluster/tfang/MLG/task_1/utils # just in case python is not smart enough 

python /home/cluster/tfang/MLG/task_1/utils/DeepHistone_runner.py --prefix "basic-model-" --use_seq True --left_flank_size 1000 --histone_bin_size 100 --conv_ksize 9 --tran_ksize 4 --batchsize 30 --epochs 10

# use sbatch on HPC


# or local on generic
# python /home/cluster/tfang/MLG/task_1/utils/DeepHistone_runner.py --prefix "basic-model-" --use_seq False --left_flank_size 1000 --histone_bin_size 100 --conv_ksize 9 --tran_ksize 4 --batchsize 30 --epochs 1
