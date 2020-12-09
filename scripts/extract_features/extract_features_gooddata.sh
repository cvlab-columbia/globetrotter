#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 NCCL_LL_THRESHOLD=4 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9995 \
--nproc_per_node=1 \
main.py \
--dataset_path [/path/to/datasets/dir] \
--results_dir [/path/to/results/dir] \
--checkpoint_dir [/path/to/checkpoints/dir] \
--runs_dir [/path/to/runs/dir] \
--dataset_info_path [/path/to/dataset/info/dir] \
--batch_size 128 \
--tokenizer_type huggingface \
--name extract_features \
--lambda_visual_loss 0 \
--lambda_orthogonality_loss 0 \
--config_data all-lang_test-zh-en \
--workers 20 \
--language_split training \
--fp16 \
--resume \
--resume_name train_globetrotter \
--test \
--test_name extract_features \
--test_options val \
--prob_predict_token 0
