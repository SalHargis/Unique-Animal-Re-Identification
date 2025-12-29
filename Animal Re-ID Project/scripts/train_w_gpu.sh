#!/bin/bash
#SBATCH -A PAS3162              
#SBATCH -J GlueOutputTest           # Job name
#SBATCH -t 25:00:00              # Time limit
#SBATCH -N 1                    # Request 1 node
#SBATCH --gpus-per-node=1       # Request 1 GPU
#SBATCH --mem=32GB              # Memory request: 32GB should be sufficient for inference
#SBATCH -p gpu                  # Use the  GPU
#SBATCH -o gluelight_miew100.%j.out  # Output file 
#SBATCH -e gluelight_miew100.%j.err  # Error file
#SBATCH --cpus-per-task=4

module load cuda/11.8.0 
source /users/PAS3162/salhargis/anaconda3/bin/activate
conda activate cv_env
cd /fs/ess/PAS3162/Kannally_Hargis_ess/code/ # py code folder
python -u glue100.py # run py
