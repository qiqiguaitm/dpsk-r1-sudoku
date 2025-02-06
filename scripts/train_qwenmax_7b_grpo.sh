#set -x
#----------------------------------------------------
BASE_MODEL_NAME=Qwen2.5-7B-Instruct-1M
MODEL_PATH=${WORK_DIR}/Qwen/${BASE_MODEL_NAME}
#----------------------------------------------------

export WORK_DIR="/nas/nfs/ofs-902-1/pnc/huggingface_hub/"
cd $WORK_DIR
DATASET_NAME=$1
$DATA_DIR=$WORK_DIR/datasets/$DATASET_NAME/
sudoku_train_path=$DATA_DIR/train.parquet
sudoku_test_path=$DATA_DIR/test.parquet
train_files="['$sudoku_train_path']"
test_files="['$sudoku_test_path']"
max_prompt_length=512
max_response_length=1024
if  [ $# -eq 2 ]; then
    max_response_length=$2
fi

#----------------------------------------------------
N_GPUS=8
if [ -z "$N_NODES" ]; then
    N_NODES=1
fi
if  [ $# -eq 3 ]; then
    MODEL_PATH=$3
fi
#----------------------------------------------------
export WANDB_API_KEY=e3dbe6853df66b090caae1511be35154c005d3c1
export WANDB_MODE=offline
TS=$(date +%s)
TASK_ID=$(date +%s)
if [ $DIST_MODE -eq 1 ]; then
    TASK_ID=$MLP_TASK_ID    
else   
    TASK_ID=x-$(date +%s)
fi
PROJ_NAME=dpskr0
EXP_NAME=dpskr0-${TASK_ID}-${BASE_MODEL_NAME}-$N_NODES-$N_GPUS-$TS
EXP_NAS_DIR=$WORK_DIR/EXP_NAS_DIR/$EXP_NAME/
export WANDB_DIR=$EXP_NAS_DIR
mkdir -p $EXP_NAS_DIR
#----------------------------------------------------


echo "----------------------------------------------------"
echo $TASK_ID,$PROJ_NAME
echo $EXP_NAME
echo $EXP_NAS_DIR
echo $train_files
echo $test_files
echo $max_prompt_length, $max_response_length
echo $LOG_DIR, $WANDB_DIR
echo $N_NODES, $N_GPUS
echo "----------------------------------------------------"

export VLLM_ATTENTION_BACKEND=XFORMERS


#1机8卡
#model_parallel_size:4,gpu_memory_utilization=0.6 
#enable_gradient_checkpointing:True,ref.fsdp_config.param_offload:True
### model:7b,max_response_length:8192,train/val_batch:16 =>ok
### model:7b,max_response_length:8192,train/val_batch:64 =>no
### model:7b,max_response_length:8192,,train/val_batch:32 =>no

#model_parallel_size:4,gpu_memory_utilization=0.6 
#model:7b,max_response_length:8192,train/val_batch:16 
### enable_gradient_checkpointing:False,ref.fsdp_config.param_offload:False =>no
### enable_gradient_checkpointing:False,ref.fsdp_config.param_offload:True =>no


#model_parallel_size:2,gpu_memory_utilization=0.6 
#enable_gradient_checkpointing:True,ref.fsdp_config.param_offload:True
### model:7b,max_response_length:8192,train/val_batch:16 =>ok


#model_parallel_size:1,gpu_memory_utilization=0.6 
#enable_gradient_checkpointing:True,ref.fsdp_config.param_offload:True
### model:7b,max_response_length:8192,train/val_batch:16 => ok


#4机32卡
#model_parallel_size:2,gpu_memory_utilization=0.6 
#enable_gradient_checkpointing:True,ref.fsdp_config.param_offload:True
### model:7b,max_response_length:8192,train/val_batch:16 =>no


python3 -u  -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((16 * N_NODES ))\
    data.val_batch_size=$((16 * N_NODES )) \
    data.max_prompt_length=$max_prompt_length\
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=64 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$EXP_NAS_DIR \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=5 2>&1 | tee $EXP_NAS_DIR/dpsk-zero.log


