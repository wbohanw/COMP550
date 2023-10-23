python train_sup.py --data_dir ./dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--train_file train_revised.json \
--dev_file dev_revised.json \
--test_file test_revised.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30 \
--seed 233 \
--num_class 97 \
--loss_type ATL \
--save_name bert_ATL_neg0.7_seed=233 \
--proj_name neg_sampling \
--run_name bert_ATL_neg0.7_seed=233 \
--neg_sample_rate 0.7 \
--cuda_device 0
