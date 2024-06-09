#!/bin/bash
#SBATCH -c 2                      # number of cores (treats)
#SBATCH --mem 32G
# #SBATCH --mem-per-cpu 32G
#SBATCH --gres=gpu:L40:1        
#SBATCH --mail-user=eitangronich@campus.technion.ac.il
#SBATCH --mail-type=END           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="IClExperiment:$1"
#SBATCH -o ./slurm_%j.out        # stdout goes to out_job.txt
#SBATCH -e ./slurm_%j.out        # stderr goes to err_job.txt

source .venv/bin/activate

python experiment.py --experiment-type $1