torchrun --nproc_per_node=8  \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=20023 sat/dist_inf_text_file.py \
    --base "sat/configs/stage1.yaml" \
    --second "sat/configs/stage2.yaml" \
    --inf-ckpt  ./checkpoints/stage1.pt \
    --inf-ckpt2 ./checkpoints/stage2.pt \
    --input-file ./example.txt \
    --output-dir ./vis_270p_1080p_example
