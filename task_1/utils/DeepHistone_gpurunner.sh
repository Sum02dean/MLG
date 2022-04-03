#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8  #8
#SBATCH --mem=100G   #50G
#SBATCH --time=24:00:00  #48:00:00
#SBATCH --output /net/cephfs/shares/von-mering.imls.uzh/tao/MLG/task_1/data/slurm_reports/output_%A_%a.txt
#SBATCH --error /net/cephfs/shares/von-mering.imls.uzh/tao/MLG/task_1/data/slurm_reports/error_%A_%a.txt

# chmod +x 
#module load generic #this should be outside this for sciencecluster to decide which particiton to use 
# we neeed very large memory and many CPUs so 
module load volta

module load volta cuda/11.2 

module load volta nvidia/cuda11.2-cudnn8.1.0


source activate /data/tfang/conda-envs/py309_MLG

cd /net/cephfs/shares/von-mering.imls.uzh/tao/MLG/task_/utils # just in case python is not smart enough 


#python DeepHistone_runner.py --model_prefix "basic-model-" --use_seq True --left_flank_size 1000 --histone_bin_size 20 --conv_ksize 9 --tran_ksize 4 --batchsize 30 --epochs 3
python DeepHistone_runner.py --model_prefix "basic-model-" --use_seq False --left_flank_size 1000 --histone_bin_size 20 --conv_ksize 9 --tran_ksize 4 --batchsize 30 --epochs 30
