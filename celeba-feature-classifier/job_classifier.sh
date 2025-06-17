#!/bin/bash
#SBATCH --job-name=attrcls      # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --nodelist=gpu01,gpu02,gpu03,gpu04
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=2       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=72G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=48:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end         # send mail when job ends
#SBATCH --mail-type=fail        # send mail if job fails
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --output=/media02/lhthai/logs/%j-%x.out
#SBATCH --error=/media02/lhthai/logs/%j-%x.err

# Load cuda
spack load cuda@11.8

# Config conda
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate gan

# Change dir
cd /media02/lhthai/Recognition-and-Classification-of-Facial-Attributes

chmod +x ./job_classifier.sh

# ./run_classifier.sh
./run_classifier_sort.sh


