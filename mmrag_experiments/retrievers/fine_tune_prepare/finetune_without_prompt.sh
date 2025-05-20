export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# 	--model_name_or_path BAAI/bge-large-en-v1.5 \  Alibaba-NLP/gte-large-en-v1.5
#      --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \

torchrun --nproc_per_node 4 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path Alibaba-NLP/gte-large-en-v1.5 \
    --cache_dir ./cache/model \
    --trust_remote_code True \
    --train_data ./finetune_data_minedHN.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 128 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_steps 1000 \
    --query_instruction_for_retrieval '' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
	--output_dir ./base_1_epoch_HN_gte-large-en-v1.5 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div