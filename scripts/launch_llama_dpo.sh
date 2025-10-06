#!/bin/bash
#SBATCH --account=lin491
#SBATCH --gres=gpu:2
#SBATCH --partition=a10
#SBATCH --time=00:07:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=llama-dpo
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --qos=standby

BETA=$1
LR=$2

# Function to find an available port
find_free_port() {
    local port
    while true; do
        # Generate a random port number between 20000 and 65000
        port=$(shuf -i 29500-29510 -n 1)
        # Check if the port is in use
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

# Function to initialize the environment and print diagnostic information
# very important that this is run within srun for training to work!!!
init_env() {
    # Load necessary modules (adjust as needed for your system)
    module load conda
    module load cuda/12.1.1
    conda activate halos

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_ON_NODE"

    # Tweaks
    export WANDB_MODE=offline
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1

    # Hugging face
    hf_cache=./.hf_cache
    hf_key=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

    export TRANSFORMERS_CACHE=${hf_cache}/hub
    export HF_HOME=${hf_cache}/
    export HF_DATASETS_CACHE=${hf_cache}/datasets

    export HUGGING_FACE_API_KEY=${hf_key}
    huggingface-cli login --token $HUGGING_FACE_API_KEY
}

export -f find_free_port
export -f init_env


# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
export MODEL_PATH=meta-llama/Llama-3.2-1B
export CKPT=/scratch/gilbreth/pmatlia/LLM_research/HALOs/data/models/Llama3-3.2-1B-dpo-${BETA}-${LR}/FINAL

accelerate launch \
    --config_file accelerate_config/fsdp_2gpu.yaml \
    --num_processes=\$SLURM_GPUS_ON_NODE \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=dpo model=llama train_datasets=[cr] test_datasets=[cr] exp_name=llama3-3.2-1B-dpo-${BETA}-${LR} \
    ++cache_dir=/scratch/gilbreth/pmatlia/LLM_research/HALOs/data/models \
    ++model.use_peft=true \
    ++model.name_or_path=\$MODEL_PATH \
    ++lr=${LR} \
    ++loss.beta=${BETA} \
    ++model.batch_size=8 ++model.gradient_accumulation_steps=4 ++model.eval_batch_size=8 \
    ++intermediate_checkpoints=false
"
# eval was removed