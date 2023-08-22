#! /bin/sh

#SBATCH --job-name=train_yelp_pytorch
#SBATCH --output=/home/yandex/MLW2023/jg/TAU-Workshop/output.out # redirect stdout
#SBATCH --error=/home/yandex/MLW2023/jg/TAU-Workshop/error_output.err # redirect stderr
#SBATCH --partition=studentrun # (see resources section)
#SBATCH --time=120 #max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=2 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=2 # CPU cores per process
#SBATCH --gpus=8 # GPUs in total

python fine_tune_native_pytorch.py
