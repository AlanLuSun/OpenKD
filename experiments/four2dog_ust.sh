#!/bin/bash
#SBATCH --job-name=graph         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --cpus-per-task=28      # number of CPUs (28, 56, 112, 224 for 1, 2, 4, 8 GPUs)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=vonneumann   # partition(large/normal/cpu) where you submit
#SBATCH --account=vonneumann1    # only require for multiple projects

# common paths: /home/changshenglu, /project/vonneumann1/changsheng, /scratch/vonneumann1/changsheng
# interactive job: srun --partition vonneumann --account=vonneumann1 --nodes=1 --gpus-per-node=1 --cpus-per-task=28 --pty bash
# login to a running job: srun --jobid=258922 -w dgx-52 --overlap --pty bash -i

# we install environments ...

module purge  # clear environment modules inherited from submission
module load cuda11.8/blas/11.8.0 cuda11.8/fft/11.8.0 cuda11.8/toolkit/11.8.0  # need Pytorch with at least cuda11.8 
source /home/changshenglu/anaconda3/bin/activate openkd

cd '/home/changshenglu/OpenKD_for_release'
echo $(pwd)

python3 train_openkd.py --cfg_file experiments/configs/openkd.yaml

python3 eval_openkd.py --cfg_file experiments/configs/openkd.yaml
python3 eval_diverse_prompts.py --cfg_file experiments/configs/openkd.yaml
python3 eval_diverse_prompts_noparsing.py --cfg_file experiments/configs/openkd.yaml




