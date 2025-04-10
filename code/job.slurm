#!/bin/bash
#SBATCH --job-name=llama-1b-sft     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=tmenezes@princeton.edu

module purge
module load anaconda3/2024.2

conda activate mt-ind

# Set number of threads for PyTorch
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

python remote-trainer-llama-1b.py