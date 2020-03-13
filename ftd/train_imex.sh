#!/bin/bash

#python train_ner.py --model "en_core_web_sm" --output-dir "/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_en_1"
#python train_new_entity_type.py --model "/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_en_1" --output-dir "/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_en_2"

python train_ner.py --model "pt_core_news_sm" --output-dir "/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_pt_1"
#python train_new_entity_type.py --model "/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_en_1" --output-dir "/home/erickj/Documents/data_f/CODE/spaCy/examples/training/model_imex_en_2"
