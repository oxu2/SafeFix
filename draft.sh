conda activate hibug

CUDA_VISIBLE_DEVICES=1 nohup python -u resnet18.py  > ./logs/test_resnet18.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u resnet18.py  > ./logs/B64_resnet18.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python stat.py