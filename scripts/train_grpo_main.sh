set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#----------------------------------------------------
BASE_MODEL=${1:-Qwen2.5-7B-Instruct-1M} #Qwen2.5-7B-Instruct-1M,Qwen2.5-14B-Instruct-1M,DeepSeek-R1-Distill-Qwen-32B
MAX_RESP_LEN=${2:-4096}  #default:4096
TASK_ID=${3:-$MLP_TASK_ID} #default:MLP_TASK_ID
if [ -z "$TASK_ID" ]; then
    TASK_ID=x-$(date +%s)
fi
CUR_TASK=${4:-step0_boiling_simple}  #defalut:step0_boiling_simple
PRE_TASK=${5:-'none'}
TOTAL_TRAINNING_STEPS=${6:-500}

temperature=${temperature:-1}
top_p=${top_p:-1}
top_k=${top_k:--1}
N_GPUS=${N_GPUS:-8}

if [ -n "$DIST_MODE" ]; then
    N_NODES=$N_NODES
else
    N_NODES=${N_NODES:-1}
fi

#----------------------------------------------------
WORK_DIR="/nas/nfs/ofs-902-1/pnc/huggingface_hub/"
PROJ_NAME=dpskr0-cl
EXP_NAME=${PROJ_NAME}-${TASK_ID}-${BASE_MODEL}-${CUR_TASK}-${N_NODES}-${N_GPUS}
EXP_SAVE_DIR=$WORK_DIR/EXP_NAS_DIR/$EXP_NAME/
mkdir -p $EXP_SAVE_DIR
tar -zcf $EXP_SAVE_DIR/verl.tar.gz  $WORK_DIR/verl 

if [[ -z "$PRE_TASK" ]] || [[ "$PRE_TASK" == "none" ]] || [[ "$PRE_TASK" == "NONE" ]]; then
    MODEL_PATH=${WORK_DIR}/MODEL_LINKS/${BASE_MODEL}
else
    PRE_CKPT_PATH=$WORK_DIR/EXP_NAS_DIR/${PROJ_NAME}-${TASK_ID}-${BASE_MODEL}-${PRE_TASK}-${N_NODES}-${N_GPUS}/actor/
    echo $PRE_CKPT_PATH
    source $WORK_DIR/verl/scripts/funcs.sh
    MODEL_PATH=$(get_newest_dir $PRE_CKPT_PATH)
fi

if [ -d "$MODEL_PATH" ]; then
    echo "MODEL_PATH EXISTS"
else
    echo "MODEL_PATH NOT EXISTS, EXITING"
    exit 1
fi

#----------------------------------------------------

cd $WORK_DIR


if [[ $CUR_TASK == *simple* ]]; then
    #DATASET_NAME=cl_sudoku_simple_20480   ### 20480, 102400
    #max_prompt_length=512
    DATASET_NAME=cl_sudoku_tips_simple_10240  
    max_prompt_length=1054
elif [[ $CUR_TASK == *medium* ]]; then
    #DATASET_NAME=cl_sudoku_medium_20480
    #max_prompt_length=512
    DATASET_NAME=cl_sudoku_tips_medium_10240
    max_prompt_length=1054
elif [[ $CUR_TASK == *hard* ]]; then
    #DATASET_NAME=cl_sudoku_hard_20480
    #max_prompt_length=512
    DATASET_NAME=cl_sudoku_tips_hard_20480
    max_prompt_length=1054
else
    echo "DATASET_NAME ERROR, EXITING"
    exit 1
fi
DATA_DIR=$WORK_DIR/verl/datasets/$DATASET_NAME/
train_files="['$DATA_DIR/train.parquet']"
test_files="['$DATA_DIR/test.parquet']"

max_response_length=$MAX_RESP_LEN  #default 4096
if [[ $BASE_MODEL = "DeepSeek-R1-Distill-Qwen-1.5B" ]] ; then
    TP_SZ=1
    BASE_BATCH_SZ=16
    gpu_memory_utilization=0.5
elif [[ $BASE_MODEL = "Qwen2.5-7B-Instruct-1M" ]] || [[ $BASE_MODEL = "DeepSeek-R1-Distill-Qwen-7B" ]] ; then
    TP_SZ=4
    BASE_BATCH_SZ=16
    gpu_memory_utilization=0.5
elif [ $BASE_MODEL = "Qwen2.5-14B-Instruct-1M" ] || [[ $BASE_MODEL = "DeepSeek-R1-Distill-Qwen-14B" ]] ; then
    TP_SZ=8
    BASE_BATCH_SZ=8
    gpu_memory_utilization=0.5
elif [ $BASE_MODEL = "DeepSeek-R1-Distill-Qwen-32B" ]; then
    TP_SZ=8
    BASE_BATCH_SZ=8
    gpu_memory_utilization=0.5
elif [ $BASE_MODEL = "Mistral-Small-24B-Instruct-2501" ]; then
    TP_SZ=8
    BASE_BATCH_SZ=8
    gpu_memory_utilization=0.5
fi

#----------------------------------------------------
export WANDB_API_KEY=e3dbe6853df66b090caae1511be35154c005d3c1
export WANDB_MODE=offline
export WANDB_DIR=$EXP_SAVE_DIR
rm -fr $EXP_SAVE_DIR/wandb/
#----------------------------------------------------


echo "---------------EXP META INFO------------------------"
echo project_name,task_id:$PROJ_NAME,$TASK_ID 
echo experiment_name:$EXP_NAME
echo experiment_save_dir:$EXP_SAVE_DIR
echo model_path:$MODEL_PATH
echo cur_task,pre_task:$CUR_TASK,$PRE_TASK
echo train_files,test_files:$train_files,$test_files
echo prompt,response:$max_prompt_length,$max_response_length
echo wandb:$WANDB_DIR
echo nodes,gpus,tp_sz:$N_NODES, $N_GPUS, $TP_SZ
echo temperature,top_p,top_k:$temperature,$top_p,$top_k
echo total_training_steps:$TOTAL_TRAINNING_STEPS

echo "----------------------------------------------------"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


if [[ $max_response_length -le 4096 ]] || [[  $BASE_MODEL == "DeepSeek-R1-Distill-Qwen-1.5B" ]]; then
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((BASE_BATCH_SZ * N_NODES ))\
    data.val_batch_size=$((BASE_BATCH_SZ * N_NODES )) \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SZ \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=64 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_SAVE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_training_steps=$TOTAL_TRAINNING_STEPS \
    trainer.total_epochs=15 2>&1 | tee  ${EXP_SAVE_DIR}/dpsk-r0-sudoku-${TASK_ID}.log
fi


if [[ $max_response_length -eq 8192 ]] && [[  $BASE_MODEL != "DeepSeek-R1-Distill-Qwen-32B" ]]; then
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((8 * N_NODES ))\
    data.val_batch_size=$((8 * N_NODES )) \
    data.max_prompt_length=$max_prompt_length\
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SZ \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_SAVE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_training_steps=$TOTAL_TRAINNING_STEPS \
    trainer.total_epochs=15 2>&1 | tee  ${EXP_SAVE_DIR}/dpsk-r0-sudoku-${TASK_ID}.log
fi

### 16384 
if [[ $max_response_length -eq 16384 ]] && [[  $BASE_MODEL != "DeepSeek-R1-Distill-Qwen-32B" ]]; then
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((8 * N_NODES ))\
    data.val_batch_size=$((8 * N_NODES )) \
    data.max_prompt_length=$max_prompt_length\
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SZ \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_SAVE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_training_steps=$TOTAL_TRAINNING_STEPS \
    trainer.total_epochs=15 2>&1 | tee  ${EXP_SAVE_DIR}/dpsk-r0-sudoku-${TASK_ID}.log
fi
#20480
if [[ $max_response_length -eq 20480 ]] && [[  $BASE_MODEL != "DeepSeek-R1-Distill-Qwen-32B" ]]; then
temperature=1.1
top_p=1.0
top_k=-1
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((8 * N_NODES ))\
    data.val_batch_size=$((8 * N_NODES )) \
    data.max_prompt_length=$max_prompt_length\
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SZ \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_SAVE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_training_steps=$TOTAL_TRAINNING_STEPS \
    trainer.total_epochs=15 2>&1 | tee  ${EXP_SAVE_DIR}/dpsk-r0-sudoku-${TASK_ID}.log
fi

if [[ $max_response_length -eq 8192 ]] && [[  "$BASE_MODEL" == "DeepSeek-R1-Distill-Qwen-32B" ]]; then
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((8 * N_NODES ))\
    data.val_batch_size=$((8 * N_NODES )) \
    data.max_prompt_length=$max_prompt_length\
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1\
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SZ \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_SAVE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_training_steps=$TOTAL_TRAINNING_STEPS \
    trainer.total_epochs=15 2>&1 | tee  ${EXP_SAVE_DIR}/dpsk-r0-sudoku-${TASK_ID}.log
fi

if [[ $max_response_length -eq 16384 ]] && [[  $BASE_MODEL == "DeepSeek-R1-Distill-Qwen-32B" ]]; then
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$((8 * N_NODES ))\
    data.val_batch_size=$((8 * N_NODES )) \
    data.max_prompt_length=$max_prompt_length\
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1\
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SZ \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${EXP_SAVE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_training_steps=$TOTAL_TRAINNING_STEPS \
    trainer.total_epochs=15 2>&1 | tee  ${EXP_SAVE_DIR}/dpsk-r0-sudoku-${TASK_ID}.log
fi


