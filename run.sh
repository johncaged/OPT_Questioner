CUDA_VISIBLE_DEVICES=4,6,7 python -m torch.distributed.launch --nproc_per_node=3 --master_port 32711 train.py
# CUDA_VISIBLE_DEVICES=4 python predict.py
