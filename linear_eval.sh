#!/bin/bash
#SBATCH --job-name=li_eval
#SBATCH --account=asignal   #project_2003370
#SBATCH --output=./err_out_li_eval/out_task_number_%A_%a.txt
#SBATCH --error=./err_out_li_eval/err_task_number_%A_%a.txt

#SBATCH --time=00-36:00:00
##SBATCH --time=00-00:15:00
#SBATCH --begin=now
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
##SBATCH --mem-per-cpu=8000

#SBATCH --nodes=1
#SBATCH --partition=gpusmall
##SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1
#SBATCH --array=7-10
##SBATCH --gres=gpu:a100:4
##SBATCH --exclude=g5102,g5201,g5301,g6101,g6102,g6201,g6301
module load pytorch/1.11
echo $SLURM_ARRAY_TASK_ID

python main_train.py -p config/params_supervised_lineval.yaml #&> logs/output_lineval.out
