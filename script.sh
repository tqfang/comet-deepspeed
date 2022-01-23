# google/t5-3B training, on 2080Ti (11GB)
deepspeed --include localhost:0,1 --master_port 30000 models/comet_seq2seq.py \
    --deepspeed deepspeed/ds_config_zero2.json \
    --model_name_or_path google/t5-xl-lm-adapt \
    --do_train \
    --train_file data/kg/atomic2020_data-feb2021/train.csv \
    --source_prefix "" \
    --output_dir data/models/comet/t5_xl_s2_bs32_fp16 \
    --overwrite_output_dir \
    --gradient_accumulation_steps=1 \
    --per_device_train_batch_size=16 \
    --max_source_length 16 \
    --max_target_length 18 \
    --text_column head_event --summary_column tail_event \
    --save_strategy epoch \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --fp16

# infer, on 2080Ti (11GB)
deepspeed --include localhost:5 --master_port=29800 models/comet_t5/comet_t5.py \
    --deepspeed deepspeed/ds_config_zero3.json \
    --model_name_or_path data/models/comet/t5_xl_s2_bs32_fp16/checkpoint-100944 \
    --do_predict \
    --train_file data/kg/few_shot_comet/csv/train_10.csv \
    --test_file system_eval/test_t5.csv \
    --source_prefix "" \
    --output_dir data/models/comet/t5_xl_s2_bs32_fp16/checkpoint-100944/output_test_2 \
    --max_source_length 16 \
    --max_target_length 18 \
    --text_column head_event \
    --summary_column tail_event \
    --num_beams 10 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate