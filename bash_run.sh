#!/bin/bash

#EMO_LIST_0=("amusement" "awe" "contentment" "excitement")
#GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
GPU_PAIRS=("0,1")
# GPU id

SCRIPT_PATH="./run_emostory.py"
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

cleanup() {
    echo "Ctrl+C，All child processes are being terminated......"
    pkill -P $$
    exit 1
}
trap cleanup SIGINT SIGTERM

STORY_DIR="./results/EmoStory_Script"
# ------------------------------
# 函数1：启动一个 emotion 列表
# ------------------------------
run_emo_list() {
    local num=${#GPU_PAIRS[@]}

    for i in "${!GPU_PAIRS[@]}"; do
        local GPU_PAIR=${GPU_PAIRS[$i]}

        CUDA_VISIBLE_DEVICES=${GPU_PAIR} \
        python ${SCRIPT_PATH} \
            --is_enhance_elements \
            --save_cross_attn_weights \
            --boost_factor=1.5 \
            --agent_result_path=${STORY_DIR} \
            --worker_id=${i} \
            --worker_num=${num} \
            &

        sleep 1
    done

    # --save_cross_attn_weights \
    # > ${LOG_DIR}/run_${EMO}.log 2>&1
    wait
}

run_emo_list
