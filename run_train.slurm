#! /bin/sh

#SBATCH --job-name=train_sarcasm_pytorch_gpus
#SBATCH --output=/home/yandex/MLW2023/mikahurvits/TAU-Workshop/dist_output.out # redirect stdout
#SBATCH --error=/home/yandex/MLW2023/mikahurvits/TAU-Workshop/edist_error_output.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=900 #max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=2 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=2 # CPU cores per process
#SBATCH --gpus=8 # GPUs in total

export HF_HOME=/home/yandex/MLW2023/mikahurvits/cache_mika3
export TRANSFORMERS_CACHE=/home/yandex/MLW2023/mikahurvits/cache_mika3
export HF_DATASETS_CACHE=/home/yandex/MLW2023/mikahurvits/cache_datasets

python sarcasm_finetunning_trainer_with_data_preparation.py