nohup ./test_combined_al.sh --combined_ratio 0.9 > combined_al_0.9.log 2>&1 < /dev/null &

nohup ./test_combined_al.sh --combined_ratio 0.7 > combined_al_0.7.log 2>&1 < /dev/null &

nohup ./test_combined_al.sh --combined_ratio 0.5 > combined_al_0.5.log 2>&1 < /dev/null &

pkill -u oxx220000 -f python

nohup ./test_al_strategies.sh --sub_ratio 0.9 > test_al_strategies_ratio0.9.log 2>&1 < /dev/null &

nohup ./test_al_strategies.sh --sub_ratio 0.7 > test_al_strategies_ratio0.7.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --al_initial_labeled 400 \
  --al_embedding_dim 256 \
  --sub_ratio 0.5 \
  > logs_imagenet/AL_resnet_imagenet_balanced_1000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --al_initial_labeled 200 \
  --al_embedding_dim 128 \
  --sub_ratio 0.5 \
  > logs_imagenet/AL_resnet_imagenet_lightweight_1000.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u alpred.py \
  --dataset imagenet --model resnet --num_added_img 1000 \
  --attributes pinkcolor_fabrictexture --num_epochs 30 \
  --use_active_learning False \
  --sub_ratio 0.5 \
  > logs_imagenet/AL_resnet_imagenet_random_1000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup ./run_active_learning.sh coreset resnet 5 > logs/al.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup ./run_active_learning.sh coreset resnet 5 > logs/coreset.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup ./run_active_learning.sh coreset resnet 5 > logs/coreset_imagenet.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup ./run_active_learning.sh coreset resnet 5 > logs/coreset_imagenet_5.log 2>&1 < /dev/null &

# move

mv logs_imagenet/RH_BS_SE_*_celeba_*.log logs/

CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py \
  --attributes orangecolor_fabrictexture \
  --dataset imagenet \
  --lvlm llava-7b \
  > logs/filter/filter_orangecolor_fabrictexture_llava.log 2>&1 < /dev/null &

#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py \
  --dataset imagenet --model clip --num_added_img 100 \
  --save_pred_model best --attributes orangecolor_fabrictexture \
  --lvlm llava --num_epochs 15 \
  > logs_imagenet/OC_FT_clip_imagenet_100.log 2>&1 < /dev/null &&

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py \
  --dataset imagenet --model clip --num_added_img 500 \
  --save_pred_model best --attributes orangecolor_fabrictexture \
  --lvlm llava --num_epochs 15 \
  > logs_imagenet/OC_FT_clip_imagenet_500.log 2>&1 < /dev/null &&

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py \
  --dataset imagenet --model clip --num_added_img 1000 \
  --save_pred_model best --attributes orangecolor_fabrictexture \
  --lvlm llava --num_epochs 15 \
  > logs_imagenet/OC_FT_clip_imagenet_1000.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py \
  --dataset imagenet --model clip --num_added_img 1000 \
  --save_pred_model best --attributes orangecolor_fabrictexture \
  --lvlm llava --num_epochs 15 \
  > logs_imagenet/OC_FT_clip_imagenet_llava_1000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --dataset imagenet --model vit --num_added_img 100  --save_pred_model best --attributes pinkcolor_fabrictexture --lvlm llava  --num_epochs 15   > logs_imagenet/PC_FT_vit_imagenet_100.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --dataset imagenet --model vit --num_added_img 500  --save_pred_model best --attributes pinkcolor_fabrictexture --lvlm llava  --num_epochs 15   > logs_imagenet/PC_FT_vit_imagenet_500.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --dataset imagenet --model vit --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --lvlm llava  --num_epochs 15   > logs_imagenet/PC_FT_vit_imagenet_1000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset celeba --model vit --num_added_img 1000  --save_pred_model best --attributes redhair_brownskin_sademotion --num_epochs 15  --seed 12376  > logs/RH_BS_SE_vit_celeba_1000.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset celeba --model vit --num_added_img 5000  --save_pred_model best --attributes redhair_brownskin_sademotion --num_epochs 15  --seed 12376  > logs/RH_BS_SE_vit_celeba_5000.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset celeba --model vit --num_added_img 10000  --save_pred_model best --attributes redhair_brownskin_sademotion --num_epochs 15  --seed 12376  > logs/RH_BS_SE_vit_celeba_10000.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --lvlm llava  > logs_imagenet/PC_FT_resnet_imagenet_llava_1000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 500  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --lvlm llava  > logs_imagenet/PC_FT_resnet_imagenet_llava_500.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --lvlm llava  > logs_imagenet/PC_FT_resnet_imagenet_llava_100.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset celeba --model resnet --num_added_img 1000  --save_pred_model best --attributes redhair_brownskin_sademotion --num_epochs 20   > logs_imagenet/RH_BS_SE_resnet_celeba_1000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset celeba --model resnet --num_added_img 5000  --save_pred_model best --attributes redhair_brownskin_sademotion --num_epochs 20 --seed 12376  > logs_imagenet/RH_BS_SE_resnet_celeba_5000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset celeba --model resnet --num_added_img 10000  --save_pred_model best --attributes redhair_brownskin_sademotion --num_epochs 20 --seed 12376  > logs_imagenet/RH_BS_SE_resnet_celeba_10000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u filter.py \
  --attributes pinkcolor_fabrictexture \
  --dataset imagenet \
  --lvlm llava-7b \
  > logs/filter/filter_pinkcolor_fabrictexture_llava.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --attributes pinkcolor_fabrictexture \
  --dataset imagenet \
  --lvlm qwen \
  > logs/filter/filter_pinkcolor_fabrictexture_qwen.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u filter.py \
  --max_memory 23 \
  --attributes redhair_brownskin_sademotion \
  > logs/filter/filter_RH_BS_SM.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 23 \
  --attributes redhair \
  --stop_at 5000 \
  > logs/filter/filter_RH.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u controlnet.py --attributes redhair_brownskin_sademotion --stop_at 1000 > logs/controlnet_redhair_brownskin_sademotion.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 500 --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 1  --gen_only 1 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_500_gen_only.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 100 --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 1  --gen_only 1 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_100_gen_only.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000 --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 1  --gen_only 1 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_gen_only.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio0.3.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0.5  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio0.5.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0.9  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio0.9.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 1  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio1.0.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0.8  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio0.8.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0.7  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio0.7.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 500  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_500_sub_ratio0.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_sub_ratio0.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 --sub_ratio 0  > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_100_sub_ratio0.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 python predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model last --attributes pinkcolor_fabrictexture --num_epochs 30

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_100.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model best --attributes purplecolor --num_epochs 30 > logs_imagenet/base_resnet_imagenet_best.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --attributes purplecolor --num_epochs 30 > logs_imagenet/base_resnet_imagenet.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_best.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model last --attributes pinkcolor_fabrictexture --num_epochs 30 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_last.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_best.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model last --attributes pinkcolor_fabrictexture --num_epochs 30 > logs_imagenet/pinkcolor_fabrictexture_resnet_imagenet_1000_last.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u lpips.py --dataset imagenet > logs_imagenet/lpips_imagenet.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a redhair_brownskin_sademotion -d 0 \
  > logs/batch/batch_resnet_redhair_brownskin_sademotion_skipfeasible.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 \
  > logs/batch/batch_resnet_redhair_brownskin.log 2>&1 < /dev/null &


  
nohup ./batch.sh -m vit -a redhair_brownskin -d 3 -g \
  > logs/g_vit_redhair_brownskin.log 2>&1 < /dev/null &


nohup ./batch.sh -m clip -a simple -d 2 -u true \
  > logs/hibug_clip_simple.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a yellowcolor -d 1 \
  > logs/batch/batch_clip_yellowcolor.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a yellowcolor -d 3 \
  > logs/batch/batch_resnet_yellowcolor.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a purplecolor -d 0 \
  > logs/batch/batch_vit_purplecolor.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a pinkcolor -d 2 \
  > logs/batch/batch_vit_pinkcolor.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a yellowcolor -d 3 \
  > logs/batch/batch_vit_yellowcolor.log 2>&1 < /dev/null &


nohup ./batch.sh -m clip -a pinkcolor -d 2 -s \
  > logs/batch/simple_batch_clip_pinkcolor.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a pinkcolor -d 1 \
  > logs/batch/batch_clip_pinkcolor.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last > logs_imagenet/base_vit_p.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --attributes purplecolor --num_epochs 30 > logs_imagenet/base_resnet.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a pinkcolor -d 2 -s \
  > logs/batch/simple_batch_resnet_pinkcolor.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a pinkcolor -d 3 \
  > logs/batch/batch_resnet_pinkcolor.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a purplecolor -d 2 -s \
  > logs/batch/simple_batch_resnet_purplecolor.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a purplecolor -d 3 \
  > logs/batch/batch_resnet_purplecolor.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 python predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model last --attributes purplecolor --num_epochs 10 

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --attributes purplecolor --num_epochs 30 > logs_imagenet/base_resnet.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 python predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model last --num_epochs 10 


CUDA_VISIBLE_DEVICES=2 python predict.py --dataset imagenet --model resnet --num_added_img 100  --save_pred_model last --num_epochs 10 

CUDA_VISIBLE_DEVICES=3 python filter.py


CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py \
  --attributes pinkcolor \
  --max_memory 22 \
  > logs/filter/filter_pinkcolor.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u filter.py \
  --attributes pinkcolor \
  > logs/filter/filter_pinkcolor.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py \
  --max_memory 23 \
  --attributes redhair_brownskin_sademotion \
  > logs/filter/filter_redhair_brownskin_sademotion.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh sd15.py --num_added_img 1113 > logs/sd_simple_imagenet_1113.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup bash time_run.sh sd15.py --num_added_img 1113 > logs/sd_simple_imagenet_1113.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 15 --pretrained 1  > logs/base_vit_pretrain_15.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 --pretrained 1  > logs/base_vit_pretrain.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a yellowhair_brownskin -d 1 \
  > logs/batch/batch_resnet_yellowhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a yellowhair_brownskin -d 3 \
  > logs/batch/batch_clip_yellowhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a yellowhair_brownskin -d 0 \
  > logs/batch/batch_vit_yellowhair_brownskin.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 150 --plot 1  > logs/base_imgnet_clip_5e-3_150.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup  python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 50 --plot 1  > logs/base_imgnet_resnet_50.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 70 --plot 1  > logs/base_imgnet_clip_5e-3_70.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1  > logs/base_imgnet_clip_1e-3.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1  > logs/base_imgnet_clip_3e-6.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1 --pretrained 1 > logs/base_imgnet_clip_pretrain.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a yellowhair_brownskin -d 2 \
  > logs/batch/batch_clip_yellowhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a yellowhair_brownskin -d 3 \
  > logs/batch/batch_resnet_yellowhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a yellowhair_brownskin -d 1 \
  > logs/batch/batch_vit_yellowhair_brownskin.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 > logs/vit.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup  python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 40 > logs/imgnet_resnt.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 40 > logs/imgnet_clip_40.log 2>&1 < /dev/null &

TODO:

change labels to 0, 1, 2, 3 in blip.py


nohup ./batch.sh -m clip -a yellowhair_brownskin -d 0 -n 10000 \
  > logs/batch/test.log 2>&1 < /dev/null &


nohup ./batch.sh -m clip -a simple -d 0 -s \
  > logs/hibug_clip_simple.log 2>&1 < /dev/null &


nohup ./batch.sh -m clip -a redhair_brownskin -d 2 -g -s \
  > logs/g_simple_clip_redhair_brownskin.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup  python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 30 > logs/imagenet_resnet.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 40 > logs/imagenet_resnet_40.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --model resnet --num_added_img 10  --save_pred_model best --num_epochs 1 --attributes redhair_brownskin_sademotion > logs/1_resnet.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py --model resnet --num_added_img 5230  --save_pred_model best --num_epochs 1 --attributes yellowhair --gen_image_dir exampleData/yellowhair > logs/y.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair_brownskin_sademotion -d 0 \
  > logs/batch/batch_clip_redhair_brownskin_sademotion.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair_brownskin_sademotion -d 0 \
  > logs/batch/batch_resnet_redhair_brownskin_sademotion.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair_brownskin_sademotion -d 1 \
  > logs/batch/batch_vit_redhair_brownskin_sademotion.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py \
  --max_memory 23 \
  --attributes redhair_brownskin_sademotion \
  > logs/filter/filter_redhair_brownskin_sademotion.log 2>&1 < /dev/null &

conda activate controlnet
CUDA_VISIBLE_DEVICES=2 nohup python -u controlnet.py --attributes yellowhair_brownskin > logs/control_yellowhair_brownskin.log 2>&1 < /dev/null &
find yellowhair_brownskin -type f -iname '*.jpg' | wc -l  

nohup ./batch.sh -m blip -a redhair -d 1 \
  > logs/batch/batch_blip_redhair.log 2>&1 < /dev/null &



nohup ./batch.sh -m clip -a redhair_sademotion -d 2 \
  > logs/batch/batch_clip_redhair_sademotion.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair_sademotion -d 1 \
  > logs/batch/batch_resnet_redhair_sademotion.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair_sademotion -d 3 \
  > logs/batch/batch_vit_redhair_sademotion.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 python predict.py --dataset imagenet --num_added_img 0  --save_pred_model last --num_epochs 40

CUDA_VISIBLE_DEVICES=1 nohup  python -u blip.py > logs/blip.log 2>&1 < /dev/null &

/data/oxx220000/imagenet_train


CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 > logs/vit.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup  python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 30 > logs/imgnet_resnt.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 40 > logs/imgnet_clip_40.log 2>&1 < /dev/null &

python count_and_split.py --data-dir /people/cs/o/oxx220000/HiBug/imagenet7
No train/val dirs found; showing 80:20 split suggestion.

n02791124        total: 1300   


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 23 \
  --attributes redhair_sademotion \
  > logs/filter/filter_redhair_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a brownskin -d 0 \
  > logs/batch/batch_clip_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a brownskin -d 0 \
  > logs/batch/batch_resnet_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a brownskin -d 1 \
  > logs/batch/batch_vit_brownskin.log 2>&1 < /dev/null &


nohup ./batch.sh -m clip -a sademotion -d 2 \
  > logs/batch/batch_clip_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a sademotion -d 0 \
  > logs/batch/batch_resnet_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a sademotion -d 3 \
  > logs/batch/batch_vit_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a sademotion -d 3 \
  > logs/batch/batch_resnet_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair -d 2 -g \
  > logs/g_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair_brownskin -d 0 -g \
  > logs/g_clip_redhair_brownskin.log 2>&1 < /dev/null &

python g_log_compare.py    \
  --prefix g_simple \
  --model clip \
  --attributes redhair

python g_log_compare.py    \
  --prefix g_simple \
  --model clip \
  --attributes redhair_brownskin

CUDA_VISIBLE_DEVICES=0 python predict.py \
  --attributes sademotion \
  --model resnet 

nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python -u predict.py \
  --attributes sademotion \
  --model resnet' \
  > logs/train_sademotion_resnet.log 2>&1 &
nohup ./batch.sh -m clip -a redhair_brownskin -d 2 -g -s \
  > logs/g_simple_clip_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 3 -g -s \
  > logs/g_simple_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair_brownskin -d 2 -g \
  > logs/g_clip_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 3 -g \
  > logs/g_clip_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g -s \
  > logs/g_simple_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 3 -g -s \
  > logs/g_simple_resnet_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g \
  > logs/g_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 3 -g \
  > logs/g_resnet_redhair.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g -n 10000 \
  > logs/g_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 3 -g -n 10000 \
  > logs/g_resnet_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g -n 10000 -s \
  > logs/g_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 3 -g -n 10000 -s \
  > logs/g_resnet_redhair.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g -s \
  > logs/g_simple_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 3 -g -s \
  > logs/g_simple_resnet_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g \
  > logs/g_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 3 -g \
  > logs/g_resnet_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u controlnet.py > logs/control_redhair_sademotion.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 33 \
  --attributes sademotion \
  > logs/filter/filter_sademotion.log 2>&1 < /dev/null &

❯ CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 33 \
  --attributes brownskin \
  > logs/filter/filter_brownskin2.log 2>&1 < /dev/null &
[1] 2528585
❯ CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 33 \
  --attributes sademotion \
  > logs/filter/filter_sademotion.log 2>&1 < /dev/null &
[2] 2529199


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 33 \
  --attributes yellowhair \
  > logs/filter/filter_yellowhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 33 \
  --attributes brownskin \
  > logs/filter/filter_brownskin2.log 2>&1 < /dev/null &

python compare_logs.py

nohup ./batch.sh -m vit -a redhair_brownskin -d 1 \
  > logs/batch/batch_vit_redhair_brownskin_best.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair_brownskin -d 0 \
  > logs/batch/batch_clip_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 1 \
  > logs/batch/batch_clip_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a redhair_brownskin -d 3 \
  > logs/batch/batch_vit_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair -d 2 \
  > logs/batch/batch_vit_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a simple -d 1 -u true \
  > logs/hibug_vit_simple.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a simple -d 0 -u true \
  > logs/hibug_vit_simple_last.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a simple -d 0 -u true \
  > logs/hibug_resnet_simple.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a simple -d 2 -u true \
  > logs/hibug_resnet_simple_last.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh sd15.py --num_added_img 12340 > logs/sd_simple_12340.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh sd15.py --num_added_img 11111 > logs/sd_simple_11111.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 3 -n 10000 \
  > logs/10000_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -n 5000 \
  > logs/5000_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 0 -n 10000 \
  > logs/10000_resnet_redhair.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a redhair -d 0 -u true \
  > logs/hibug_resnet_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 3 -u true \
  > logs/hibug_resnet_redhair_brownskin.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 0 \
  > logs/batch/batch_resnet_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair_brownskin -d 3 \
  > logs/batch/batch_resnet_redhair_brownskin.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 python predict.py --model resnet --num_added_img 1000 --attributes redhair 

CUDA_VISIBLE_DEVICES=2 python -u predict.py --model resnet --num_added_img 1000 --attributes redhair  > logs/test.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py \
  --dir exampleData/sad \
  --attributes sademotion \
  > logs/filter/filter_sad.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0  python filter.py --max_memory 23 --dir exampleData/redhair --attributes redhair
CUDA_VISIBLE_DEVICES=0  python filter.py --max_memory 23 --dir exampleData/sad --attributes sademotion

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh sd15.py --attributes yellowhair --num_added_img 12340 > logs/sd_yellowhair_12340.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a redhair -d 1 -u \
  > logs/hibug_vit_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a redhair_brownskin -d 2 -u true \
  > logs/hibug_vit_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 python -u predict.py --model resnet --num_added_img 1000 --hibug true --attributes redhair  > logs/test.log 2>&1 < /dev/null &

python filter_hibug_images.py --attributes redhair

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py  > logs/resnet18_no_data_added.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair -d 1 -u \
  > logs/hibug_resnet_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -u true \
  > logs/hibug_resnet_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u filter.py \
  --max_memory 23 \
  --dir hibug_redhair_brownskin \
  --attributes redhair_brownskin \
  --stop_at 15000 \
  > logs/filter/filter_hibug_redhair_brownskin_15000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py \
  --max_memory 23 \
  --dir hibug_redhair \
  --attributes redhair \
  --stop_at 15000 \
  > logs/filter/filter_hibug_redhair_12340.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh sd15.py --attributes brownskin --num_added_img 12340 > logs/sd_brownskin_12340.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 1 > logs/batch/batch_redhair_brownskin_100000.log 2>&1 < /dev/null &

nohup ./for.sh -n 10000 -a redhair -d 3 > logs/for_redhair_10000.log 2>&1 &

nohup ./for.sh -n 10000 -a redhair_brownskin -d 1 -g > logs/g_redhair_brownskin_10000.log 2>&1 &
nohup ./for.sh -n 10000 -a redhair_brownskin -d 2 > logs/for_redhair_brownskin_10000.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u filter.py \
  --dir exampleData/redhair_brownskin \
  --attributes redhair_brownskin \
  --max_memory 23 \
  --feasible_only \
  --results_file save/json/redhair_brownskin_results_04260045.json \
  > logs/filter/filter_redhairb_feasible.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u filter.py \
  --dir exampleData/redhair \
  --attributes redhair \
  --max_memory 23 \
  --feasible_only \
  --results_file save/json/redhair_results_04260005.json \
  > logs/filter/filter_redhair_feasible.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u filter.py \
  --max_memory 23 \
  --dir exampleData/redhair_brownskin \
  --attributes redhair_brownskin \
  --stop_at 15000 \
  > logs/filter/filter_redhair_brownskin_15000.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup python -u filter.py \
  --max_memory 23 \
  --dir exampleData/redhair_brownskin \
  --attributes redhair_brownskin \
  --stop_at 21111 \
  > logs/filter/filter_redhair_brownskin_21111.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py \
  --max_memory 23 \
  --dir exampleData/redhair \
  --attributes redhair \
  --stop_at 20000 \
  > logs/filter/filter_redhair_20000.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 3 > logs/batch/batch_clip_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a redhair_brownskin -d 3 > logs/batch/batch_vit_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./for.sh -n 10000 -a redhair_brownskin -d 0 > logs/for_redhair_brownskin_10000.log 2>&1 &
nohup ./for.sh -n 10000 -a redhair_brownskin -d 1 -g > logs/g_redhair_brownskin_10000.log 2>&1 &
nohup ./batch.sh -m clip -a redhair_brownskin -d 2 -g > logs/g_batch_vit_redhair_brownskin.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet --num_added_img 12340 > logs/resnet_12340.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u filter.py --max_memory 23 --dir exampleData/redhair --attributes redhair > logs/filter/filter_redhair_20000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model clip --num_added_img 0 --plot 1 > logs/clip_0imgs.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet --num_added_img 0 > logs/resnet_0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model resnet --num_added_img 0 > logs/resnet_0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model vit --num_added_img 0 > logs/vit_0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2,3 python filter.py

nohup ./for.sh -n 0 -a redhair_brownskin -d 3 > logs/for_redhair_brownskin_0.log 2>&1 &
nohup ./for.sh -n 0 -a redhair -d 2 > logs/for_redhair_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,2,3 nohup python -u filter.py --max_memory 11GiB --dir exampleData/redhair_brownskin_46703 > logs/filter/filter_redhair_brownskin_46703.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair -d 1 > logs/batch/batch_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair -d 1 > logs/batch/batch_vit_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair -d 1 -g > logs/batch/batch_resnet_redhair_g.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g > logs/resnet_redhair_brownskin_g.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair -d 2 > logs/batch/batch_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 1 > logs/batch/batch_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair -d 1 > logs/batch/batch_vit_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair -d 1 > logs/batch/batch__redhair.log 2>&1 < /dev/null &

nohup ./for.sh -n 1000 -a redhair_brownskin -d 0 > logs/for_redhair_brownskin_1000.log 2>&1 &
nohup ./for.sh -n 2000 -a redhair_brownskin -d 2 > logs/for_redhair_brownskin_2000.log 2>&1 &
nohup ./for.sh -n 4000 -a redhair_brownskin -d 3 > logs/for_redhair_brownskin_4000.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model vit > logs/vit_redhair_brownskin_20.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model resnet > logs/resnet_redhair_brownskin_20.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model clip > logs/clip_redhair_brownskin_20.log 2>&1 < /dev/null &
nohup ./for.sh > for1000.log 2>&1 &
nohup ./for.sh > for2000.log 2>&1 &
nohup ./for.sh > for4000.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model vit --num_added_img 1000 > logs/vit_redhair_brownskin_20.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model resnet --num_added_img 1000 > logs/resnet_redhair_brownskin_20.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model clip > logs/clip_redhair_brownskin_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model vit > logs/vit_redhair_brownskin_10.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model resnet > logs/resnet_redhair_brownskin_10.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model clip > logs/clip_redhair_brownskin_10.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup python -u compare_acc.py \
  --orig logs/stat_vit_origin_0421030925.log \
  --new logs/stat_vit_redhair_0421030927.log \
  > logs/compare_vit_redhair_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 \
nohup bash time_run.sh stat.py \
  --predictions predictions_vit_origin_0421030925.npy \
  > logs/stat_vit_origin_0421030925.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 \
nohup bash time_run.sh stat.py \
  --predictions predictions_vit_04080238_0421030927.npy \
  > logs/stat_vit_redhair_0421030927.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u compare_acc.py \
  --new logs/stat_origin_clip_0416163547.log \
  --orig logs/stat_redhair_clip_0417122037.log \
  > logs/compare_clip_redhair_swap.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u compare_acc.py \
  --orig logs/stat_origin_clip_0417121952.log \
  --new logs/stat_clip_redhair_0421025340.log \
  > logs/compare_clip_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u compare_acc.py \
  --orig logs/stat_clip_origin_0421030416.log \
  --new logs/stat_clip_redhair_0421025340.log \
  > logs/compare_clip_redhair_wrong.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u compare_acc.py \
  --orig logs/stat_origin_vit_04161402.log \
  --new logs/stat_redhair_vit_0416140931.log \
  > logs/compare_vit_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup python -u compare_acc.py \
  > logs/compare_resnet_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup bash time_run.sh predict.py --model clip --feasible-file a.txt > logs/clip_origin_10.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 \
nohup bash time_run.sh stat.py \
  --predictions predictions_clip_origin_0421030416.npy \
  > logs/stat_clip_origin_0421030416.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 \
nohup bash time_run.sh stat.py \
  --predictions pred/predictions_vit_origin_0421030925.npy \
  > logs/compare/predictions_vit_origin_0421030925.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 \
nohup bash time_run.sh stat.py \
  --predictions predictions_04080238_clip_0421025340.npy \
  > logs/stat_clip_redhair_0421025340.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model vit --feasible-file a.txt > logs/vit_origin_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup bash time_run.sh predict.py --model vit > logs/vit_redhair_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet18 > logs/resnet_redhair_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup bash time_run.sh predict.py --model resnet18 --feasible-file a.txt > logs/resnet_origin_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model clip  > logs/clip_redhair_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model clip --feasible-file a.txt > logs/clip_origin_20.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model clip  > logs/clip_redhair_10.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model vit --feasible-file a.txt > logs/vit_origin_15.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model vit > logs/vit_redhair_15.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model resnet18 --feasible-file a.txt > logs/resnet18_no_pretrained.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh stat.py > logs/stat_vit_no_data.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet18 > logs/resnet18_redhair_15.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup bash time_run.sh predict.py --feasible-file a.txt > logs/resnet_origin_15.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --feasible-file a.txt > logs/vit_origin_15.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --feasible-file a.txt > logs/vit_origin.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py  > logs/vit_redhair2.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model resnet18 --feasible-file a.txt > logs/resnet18_no_pretrained.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet18 > logs/resnet18_no_data_added.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py > logs/resnet18_no_data.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh stat.py > logs/red.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh stat.py > logs/red.log 2>&1 < /dev/null &



CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py > logs/vit_no_data_added.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py > logs/vit_no_data.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py > logs/resnet18_no_data_added.log 2>&1 < /dev/null &

conda activate hibug
conda activate ad

CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py  > logs/resnet18_no_data_added.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > logs/vit_redhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u predict.py  > logs/resnet18_origin_label_train.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > logs/resnet18_qwen_label.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > logs/resnet18_origin_label.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > logs/test_resnet18.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py  > logs/B64_resnet18.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 python stat.py

CUDA_VISIBLE_DEVICES=0 python filter.py

CUDA_VISIBLE_DEVICES=0,2,3 python filter.py


CUDA_VISIBLE_DEVICES=1 nohup python -u filter.py  > logs/filter.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u map.py  > logs/map.log 2>&1 < /dev/null &

# CONDA_ENV=myenv python filter.py --dir /path/to/images --file-type jpg
CONDA_ENV=control python filter.py

CUDA_VISIBLE_DEVICES=1,2,3 python filter.py

CUDA_VISIBLE_DEVICES=1,2,3 python filter.py --verbose True