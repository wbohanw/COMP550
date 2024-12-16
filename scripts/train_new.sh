#!/bin/bash


LOSS_TYPES=("balance_softmax" "HingeABL" "ATL" "AFL" "SAT" "MeanSAT")  
USE_CL_VALUES=(0)                       
TRAIN_FILE="train_revised.json"       
BASE_SAVE_NAME="docred_bert"           


for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
    for USE_CL in "${USE_CL_VALUES[@]}"; do


        CL_SUFFIX=""
        if [ "$USE_CL" -eq 1 ]; then
            CL_SUFFIX="_cl"
        fi

        SAVE_NAME="${BASE_SAVE_NAME}_${LOSS_TYPE}_new${CL_SUFFIX}"


        echo "Running with loss_type=$LOSS_TYPE, use_cl=$USE_CL, save_name=$SAVE_NAME"


        python train.py \
            --data_dir ./dataset/docred \
            --transformer_type bert \
            --model_name_or_path bert-base-cased \
            --train_file $TRAIN_FILE \
            --dev_file dev_revised.json \
            --test_file test_revised.json \
            --train_batch_size 4 \
            --test_batch_size 4 \
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
            --cuda_device 0

    done
done
