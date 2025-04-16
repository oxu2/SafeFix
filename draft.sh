CUDA_VISIBLE_DEVICES=3 nohup bash run.sh predict.py --model vit --feasible-file a.txt > ./logs/vit_origin_15.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup bash run.sh predict.py --model vit > ./logs/vit_redhair_15.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup bash run.sh predict.py --model resnet18 --feasible-file a.txt > ./logs/resnet18_no_pretrained.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup bash run.sh stat.py > ./logs/stat_vit_no_data.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash run.sh predict.py --model resnet18 > ./logs/resnet18_redhair_15.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup bash run.sh predict.py --feasible-file a.txt > ./logs/resnet_origin_15.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --feasible-file a.txt > ./logs/vit_origin_15.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --feasible-file a.txt > ./logs/vit_origin.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py  > ./logs/vit_redhair2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash run.sh predict.py --model resnet18 --feasible-file a.txt > ./logs/resnet18_no_pretrained.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash run.sh predict.py --model resnet18 > ./logs/resnet18_no_data_added.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup bash run.sh predict.py > ./logs/resnet18_no_data.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup bash run.sh stat.py > ./logs/red.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 nohup bash run.sh stat.py > ./logs/red.log 2>&1 &



CUDA_VISIBLE_DEVICES=2 nohup bash run.sh predict.py > ./logs/vit_no_data_added.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash run.sh predict.py > ./logs/vit_no_data.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash run.sh predict.py > ./logs/resnet18_no_data_added.log 2>&1 &

conda activate hibug
conda activate ad

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py  > ./logs/resnet18_no_data_added.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > ./logs/vit_redhair.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py  > ./logs/resnet18_origin_label_train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > ./logs/resnet18_qwen_label.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > ./logs/resnet18_origin_label.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > ./logs/test_resnet18.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > ./logs/B64_resnet18.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python stat.py

CUDA_VISIBLE_DEVICES=0 python filter.py

CUDA_VISIBLE_DEVICES=0,2,3 python filter.py


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py  > ./logs/filter.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u map.py  > ./logs/map.log 2>&1 &

# CONDA_ENV=myenv python filter.py --dir /path/to/images --file-type jpg
CONDA_ENV=control python filter.py
