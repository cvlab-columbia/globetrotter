#!/usr/bin/env bash
python test.py sentence_translation \
--name_model=[name_model](e.g. "train_globetrotter") \
--results_path=[/path/to/results/folder](folder where you want results to be saved) \
--extracted_features_name=[extracted_features_config-name_more-info.pth](created with scripts/extract_features/*.sh) \
--method=common \
--normalize