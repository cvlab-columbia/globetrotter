#!/usr/bin/env bash
python test.py align_words \
--name_model [name_model] \
--checkpoint_dir [/path/to/checkpoints/dir] \
--results_path [/path/to/results/dir] \
--model_type globetrotter \
--procrustes \
--dataset_info_path [/path/to/dataset/info/dir] \
--config_data all-lang_test-zh-en \
--tokenizer_type huggingface