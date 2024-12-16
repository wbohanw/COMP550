#!/bin/bash

# 定义参数列表
LOSS_TYPES=("AML" "balance_softmax" "HingeABL" "ATL" "AFL" "SAT" "MeanSAT")  # 不同的loss_type
USE_CL_VALUES=(1)                       # use_cl取值
TRAIN_FILE="train_revised.json"       # 固定的train_file
BASE_SAVE_NAME="docred_bert"            # 基础的save_name前缀

# 循环遍历不同的loss_type和use_cl组合
for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
    for USE_CL in "${USE_CL_VALUES[@]}"; do

        # 根据loss_type和use_cl动态生成save_name
        CL_SUFFIX=""
        if [ "$USE_CL" -eq 1 ]; then
            CL_SUFFIX="_cl"
        fi

        SAVE_NAME="${BASE_SAVE_NAME}_${LOSS_TYPE}_new${CL_SUFFIX}"

        # 打印当前运行的配置
        echo "Running with loss_type=$LOSS_TYPE, use_cl=$USE_CL, save_name=$SAVE_NAME"

        # 执行Python脚本
        python train.py \
            --data_dir ./dataset/docred \
            --transformer_type bert \
            --model_name_or_path bert-base-cased \
            --train_file $TRAIN_FILE \
            --dev_file dev_revised.json \
            --test_file test_revised.json \
            --train_batch_size 4 \
            --test_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --num_labels 4 \
            --learning_rate 5e-5 \
            --max_grad_norm 1.0 \
            --warmup_ratio 0.06 \
            --num_train_epochs 25 \
            --seed 66 \
            --num_class 97 \
            --loss_type $LOSS_TYPE \
            --save_name $SAVE_NAME \
            --proj_name docred \
            --run_name bert_${LOSS_TYPE} \
            --use_cl $USE_CL \
            --cuda_device 1

    done
done
