#set -x
export WORK_DIR="/nas/nfs/ofs-902-1/pnc/huggingface_hub/"
cd $WORK_DIR
base_model=$1 ##Qwen2.5-7B-Instruct-1M,Qwen2.5-14B-Instruct-1M,Mistral-Small-24B-Instruct-2501
max_response_length=$2 # 2048,4096,    8192(risk to oom)


#########testing#########
base_model=Qwen2.5-7B-Instruct-1M
max_response_length=512
#DIST_MODE=1
#MLP_TASK_ID=j-123
#########testing#########

#########starter#########
#base_model=Qwen2.5-7B-Instruct-1M
#max_response_length=8192
#########starter#########




export N_GPUS=8
if [[ -n "$DIST_MODE" ]] && [[ "$DIST_MODE" -eq 1 ]]; then
    task_id=$MLP_TASK_ID  
    if [ -n "$N_NODES" ]; then
        export N_NODES
    else
        export N_NODES=1
    fi
else   
    task_id=x-$(date +%s)
    export N_NODES=1
fi



echo "--------------step 0: all tasks boiling  ---------------------"
export temperature=1.2
export top_p=1.0
export top_k=-1 

echo "--------------simple tasks---------------------"
pre_task=''
cur_task='step0_boiling_simple'
echo $base_model $cur_task $max_response_length $task_id $pre_task
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi
echo "--------------medium tasks---------------------"
pre_task=$cur_task
cur_task=’step0_boiling_medium‘
echo $base_model $cur_task $max_response_length $task_id $pre_task
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi
echo "--------------hard tasks---------------------"
pre_task=$cur_task
cur_task='step0_boiling_hard'
echo $base_model $cur_task $max_response_length $task_id $pre_task
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi

echo "--------------step 1: hard tasks anneling ---------------------"
export temperature=1.0
export top_p=0.9
export top_k=-1 

pre_task=$cur_task
cur_task='step1_anneling_hard'
echo $base_model $cur_task $max_response_length $task_id $pre_task
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $cur_task $max_response_length  $task_id $pre_task
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi

echo "--------------step 2: hard tasks anneling ---------------------"
export temperature=0.9
export top_p=0.7
export top_k=-1 

pre_task=$cur_task
cur_task='step2_anneling_hard'
echo $base_model $cur_task $max_response_length $task_id $pre_task
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $cur_task $max_response_length $task_id $pre_task
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi
