#set -x
export WORK_DIR="/nas/nfs/ofs-902-1/pnc/huggingface_hub/"
cd $WORK_DIR
base_model=${1:-Qwen2.5-7B-Instruct-1M} ##Qwen2.5-7B-Instruct-1M,Qwen2.5-14B-Instruct-1M,Mistral-Small-24B-Instruct-2501
max_response_length=${2:-4096} # 2048,4096,    8192(risk to oom)
specific_task_id=${3:-x-99990206099999-ddl2q}    #x-20250206094445-ddl2q #x-20250206061047-v9hz8 #x-20250206054952-qn6jb
base_training_steps=${4:-300}  #default:300

#########starter#########
#base_model=Qwen2.5-14B-Instruct-1M
#max_response_length=4096
#specific_task_id=y-99990207079999-c8vkm  #y-20250207073827-c8vkm
#########starter#########


#########testing#########
#base_model=Qwen2.5-7B-Instruct-1M
#max_response_length=512
#DIST_MODE=1
#MLP_TASK_ID=j-123
#specific_task_id=z-99990207079999-jcnrg  #z-20250207073823-jcnrg
#########testing#########



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

if [ -n "$specific_task_id" ];then
   task_id=$specific_task_id
fi



echo "--------------step 0: all tasks boiling  ---------------------"
export temperature=1.2
export top_p=1.0
export top_k=-1 


echo "--------------simple tasks---------------------"
pre_task='none'
cur_task='step0_boiling_simple'
task_steps=$(awk "BEGIN {printf \"%.0f\", $base_training_steps * 0.25}")
echo base_model cur_task max_response_length task_id pre_task task_steps
echo $base_model $cur_task $max_response_length $task_id $pre_task $task_steps
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task $task_steps
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi


echo "--------------medium tasks---------------------"
pre_task='step0_boiling_simple'
cur_task='step0_boiling_medium'
task_steps=$(awk "BEGIN {printf \"%.0f\", $base_training_steps * 0.5}")
echo base_model cur_task max_response_length task_id pre_task task_steps
echo $base_model $cur_task $max_response_length $task_id $pre_task $task_steps
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task $task_steps
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi
echo "--------------hard tasks---------------------"
pre_task='step0_boiling_medium'
cur_task='step0_boiling_hard'
task_steps=$(awk "BEGIN {printf \"%.0f\", $base_training_steps * 1.0}")
echo base_model cur_task max_response_length task_id pre_task task_steps
echo $base_model $cur_task $max_response_length $task_id $pre_task $task_steps
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task $task_steps
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

pre_task='step0_boiling_hard'
cur_task='step1_anneling_hard'
task_steps=$(awk "BEGIN {printf \"%.0f\", $base_training_steps * 2.0}")
echo base_model cur_task max_response_length task_id pre_task task_steps
echo $base_model $cur_task $max_response_length $task_id $pre_task $task_steps
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task $task_steps
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

pre_task='step1_anneling_hard'
cur_task='step2_anneling_hard'
task_steps=$(awk "BEGIN {printf \"%.0f\", $base_training_steps * 1.0}")
echo base_model cur_task max_response_length task_id pre_task task_steps
echo $base_model $cur_task $max_response_length $task_id $pre_task $task_steps
bash $WORK_DIR/verl/scripts/train_grpo_main.sh $base_model $max_response_length  $task_id $cur_task  $pre_task  $task_steps
if [ $? -eq 0 ]; then
    echo  $cur_task done
else
    echo $cur_task fail
    exit 1
fi
