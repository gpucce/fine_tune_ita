#SBATCH --job-name=evaluating_summary_minerva_1B_lora            # Job name
#SBATCH -o logs/evaluating_summary_minerva_1B_lora-job.out       # Name of stdout output file
#SBATCH -e logs/evaluating_summary_minerva_1B_lora-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 08:00:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:1                    # number of gpus per node


#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

source /leonardo/home/userexternal/lmoroni0/__Work/llm_summarization_finetunig/env/bin/activate

python /leonardo/home/userexternal/lmoroni0/__Work/llm_summarization_finetunig/finetuning/evaluate_model.py


