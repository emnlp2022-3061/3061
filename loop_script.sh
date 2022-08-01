#!/bin/bash

function contrastive() {
    echo "contrastive ${1} start" >> progress.txt
    conda activate sim
    cd SimCSE
    NUM_GPU=1
    PORT_ID=$(expr $RANDOM + 1000)
    export OMP_NUM_THREADS=8
    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path ../pretrain/sim_snli/pytorch_my_loop_64_snli_bert \
    --train_file ../CLINE/data/disk/snli/snli_contrastive_${1}.csv \
    --output_dir result/my-sup-simcse-bert-base-uncased-loop-64-snlitrained \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    --overwrite_cache True \
    "$@"
    cd ..
    echo "contrastive ${1} done" >> progress.txt
}

function pretraining() {
    echo "pretraining ${1} start" >> progress.txt
    conda activate Bert
    cd pretrain/sim_snli
    rm -rf run_my_loop_64_snli
    mkdir run_my_loop_64_snli
    cd ../..
    cd bert-master
    python run_pretraining.py --input_file=../pretrain/sim_snli/create/tf_examples_${1}.tfrecord --output_dir=../pretrain/sim_snli/run_my_loop_64_snli --do_train=True --bert_config_file=../sup-simcse-bert-base-uncased/config.json --init_checkpoint=../my-sup-simcse-bert-loop-64-snlitrained/bert_base_uncased.ckpt --num_train_steps=790 --num_warmup_steps=79
    cd ..
    echo "pretraining ${1} done" >> progress.txt
    conda deactivate
}

function pt_to_tf() {
    echo "pt_to_tf ${1} start" >> progress.txt
    cd transformers/src/transformers/models/bert
    python convert_bert_pytorch_checkpoint_to_original_tf.py --cache_dir ../../../../../SimCSE/result/my-sup-simcse-bert-base-uncased-loop-64-snlitrained --pytorch_model_path ../../../../../SimCSE/result/my-sup-simcse-bert-base-uncased-loop-64-snlitrained/pytorch_model.bin --tf_cache_dir ../../../../../my-sup-simcse-bert-loop-64-snlitrained --model_name bert-base-uncased
    cd ../../../../..
    echo "pt_to_tf ${1} done" >> progress.txt
    conda deactivate
}

function tf_to_pt() {
    echo "tf_to_pt ${1} start" >> progress.txt
    conda activate tran
    transformers-cli convert --model_type bert --tf_checkpoint ./pretrain/sim_snli/run_my_loop_64_snli/model.ckpt-790 --config ./sup-simcse-bert-base-uncased/config.json --pytorch_dump_output ./pretrain/sim_snli/pytorch_my_loop_64_snli_bert/pytorch_model.bin
    echo "tf_to_pt ${1} done" >> progress.txt
    conda deactivate
}

pt_to_tf 1
pretraining 1
tf_to_pt 1

cc=2
while [ $cc -le 64 ]; do
    contrastive $cc
    pt_to_tf $cc
    pretraining $cc
    tf_to_pt $cc
    cc=`expr $cc + 1`
done
echo "All done!"