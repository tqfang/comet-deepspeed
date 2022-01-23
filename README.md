# Training COMET using seq2seq setting

Use `AutoModelForSeq2SeqLM` in Huggingface Transformers to train COMET. The codes are modified from [run_summarization.py](https://github.com/huggingface/transformers/blob/fa39ff9fc4401228601b57f5f369e9942aa5009c/examples/pytorch/summarization/run_summarization.py) in the official example codes for transformers version 4.16.0.dev0.

The `./deepspeed/` folder is copied from https://github.com/huggingface/transformers/tree/master/tests/deepspeed .

The training data of ATOMIC2020 can be downloaded at https://allenai.org/data/atomic-2020. You need to convert the .tsv file to .csv to be compatible with the dataloader in transformers.

## Dependencies

python
```
torch==1.7.1
cudatoolkit=11.0
transformers==4.15.0
deepspeed==0.5.10
```

others
```
GCC/G++ 5.2.0 (to complie deepspeed ops)
```
## Usage

### 1. Normal training without memory optimization:


```
CUDA_VISIBLE_DEVICES=0 python models/comet_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --train_file /path/to/train.csv \
    --source_prefix "" \
    --output_dir data/models/t5-small \
    --overwrite_output_dir \
    --gradient_accumulation_steps=4 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=4 \
    --max_source_length 16 \
    --max_target_length 18 \
    --text_column head_event --summary_column tail_event \
    --save_strategy epoch \
    --num_train_epochs 3 \
    --learning_rate 1e-5 
```

### 2. Train with gradient_checkpointing=True. Smaller memory usage, meanwhile lower training speed.

```
CUDA_VISIBLE_DEVICES=0 python models/comet_seq2seq.py \
    --model_name_or_path t5-small \
    --do_train \
    --train_file /path/to/train.csv \
    --source_prefix "" \
    --output_dir data/models/t5-small \
    --overwrite_output_dir \
    --gradient_accumulation_steps=4 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=4 \
    --max_source_length 16 \
    --max_target_length 18 \
    --text_column head_event --summary_column tail_event \
    --save_strategy epoch \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --gradient_checkpointing
```

### 3. Train with DeepSpeed (Either zero-stage2 or zero-stage3)

```
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
```

### 4. Comparison of memory usage of different memory optimization methods

Compare the memory usage on NVIDIA RTX A6000 (48685MB memory) and Nvidia GeForce 3090 (24268MB memory).

#### 1. fp16

T5-3B: effects of fp16. A 20\% reduce of memory size.

||Device|fp16|Batch Size x Grad-Accum x Num-GPU|Memory Usage|Time to Train a Batch|
|:--:|:--:|:--:|:--:|:--:|:--:|
|vanilla|A6000|False|8x4x1|47.5k M|1.5s/32ex|
|vanilla|A6000|True|8x4x1|31k M|1.0s/32ex|
|vanilla|3090|False|1x32x1|❌|-|-|
|vanilla|3090|True|1x32x1|❌|-|-|

#### 2. gradient_checkpointing

T5-3B: Effects of gradient_checkpointing.

||Device|fp16|Batch Size x Grad-Accum x Num-GPU|Memory Usage|Time to Train a Batch|
|:--:|:--:|:--:|:--:|:--:|:--:|
|vanilla|A6000|False|8x4x1|47k M|1.5s/32ex|
|vanilla|A6000|True|8x4x1|31k M|1.0s/32ex|
|grad-ckpt|A6000|False|8x4x1|46.4k M|1.3s/32ex|
|grad-ckpt|A6000|True|8x4x1|23.9k M|1.1/32ex|
|vanilla|3090|True|1x32x1|❌|-|
|grad-ckpt|3090|True|1x32x1|23.8k M|15s/32ex|

#### 3. Deepspeed stage 2

T5-3B: Effects of deepspeed.

||Device|fp16|Batch Size x Grad-Accum x Num-GPU|Memory Usage|Time to Train a Batch|
|:--:|:--:|:--:|:--:|:--:|:--:|
|vanilla|3090|True|1x32x1|❌|-|-|
|grad-ckpt|3090|True|1x32x1|23k M|13.5s/32ex|
|stage2|3090|True|32x1x1|20.3k M|7.5s/32ex|
|stage2|3090|True|16x1x2|20.3k M|6.36s/32ex|
|stage2|3090|True|32x1x2|20.3k M|3.75s/32ex|

#### 4. Deepspeed stage 3

stage3 will lead to smaller usage of memory but way smaller training speed.

### 5. Automatic Evaluation Result on ATOMIC2020 data

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR| ROUGE-L| CIDEr|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|T5-3B (no deepspeed), lr1e-5, epoch 3|0.346| 0.184| 0.12| 0.084| 0.19| 0.422| 0.646|
|T5-3B (no deepspeed), lr1e-5, epoch 2|0.348| 0.185| 0.121| 0.085| 0.19| 0.424| 0.651|
|T5-3B (no deepspeed), lr1e-5, epoch 1|0.343| 0.177| 0.113| 0.079| 0.186| 0.416| 0.629|
|T5-3B (ds_stage2, fp16) epoch 3|0.340| 0.182| 0.118| 0.083| 0.189| 0.418| 0.637|
|T5-3B (ds_stage2, fp16) epoch 2|0.337| 0.177| 0.114| 0.078| 0.189| 0.419| 0.633|
|T5-3B (ds_stage2, fp16) epoch 1|0.335| 0.174| 0.112| 0.076| 0.186| 0.415| 0.632|

## Useful discussions regarding environment setups

- Errors building DeepSpeed Ops: https://github.com/microsoft/DeepSpeed/issues/885

## TODO

DeepSpeed without `Trainer()`: https://huggingface.co/docs/transformers/main_classes/deepspeed#deepspeed-non-trainer-integration



