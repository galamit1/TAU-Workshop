#! /bin/sh

#SBATCH --job-name=awesome
#SBATCH --output=/specific/a/home/cc/students/csguests/jg/awesome.out # redirect stdout
#SBATCH --error=/specific/a/home/cc/students/csguests/jg/awesome.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=1 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python awesome.py
