#! /bin/sh

#SBATCH --job-name=awesome
#SBATCH --output=/home/yandex/MLW2023/jg/test.out # redirect stdout
#SBATCH --error=/home/yandex/MLW2023/jg/test.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=100 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=1 # CPU cores per process
#SBATCH --gpus=1           # Request 1 GPU


python from_saved_model.py
