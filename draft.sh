conda activate hibug
conda activate ad

CUDA_VISIBLE_DEVICES=1 nohup python -u resnet18.py  > ./logs/test_resnet18.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u resnet18.py  > ./logs/B64_resnet18.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python stat.py

CUDA_VISIBLE_DEVICES=0 python filter.py

CUDA_VISIBLE_DEVICES=0,2,3 python filter.py


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py  > ./logs/filter.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u map.py  > ./logs/map.log 2>&1 &