#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=1  #8
#SBATCH --mem=100G   #50G
#SBATCH --time=24:00:00  #48:00:00
#SBATCH --output /net/cephfs/shares/von-mering.imls.uzh/tao/alphafold_output/slurm_reports/output_%A_%a.txt
#SBATCH --error /net/cephfs/shares/von-mering.imls.uzh/tao/alphafold_output/slurm_reports/error_%A_%a.txt

# chmod +x 
#module load generic #this should be outside this for sciencecluster to decide which particiton to use 
# we neeed very large memory and many CPUs so 
module load volta

module load volta cuda/11.2 

module load volta nvidia/cuda11.2-cudnn8.1.0


source activate /data/tfang/conda-envs/py309_MLG

cd /home/cluster/tfang/MLG/task_1/utils # just in case python is not smart enough 

python /home/cluster/tfang/MLG/task_1/utils/DeepHistone_runner.py