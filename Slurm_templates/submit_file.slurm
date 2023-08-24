#! /bin/sh

#SBATCH --job-name=python_script
#SBATCH --output=/home/yandex/MLW2023/mikahurvits/example/awesome.out # redirect stdout
#SBATCH --error=/home/yandex/MLW2023/mikahurvits/example/awesome.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=1 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python python_script.py

