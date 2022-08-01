#!/bin/bash
c=1
while [ $c -le 32 ]; do
    python create_pretraining_data.py --input_file=../pretrain/acl/unlabelled_sep_$c.txt --output_file=../pretrain/sim_imdb/create/tf_examples_$c.tfrecord --vocab_file=../sup-simcse-bert-base-uncased/vocab.txt
    c=`expr $c + 1`
done
