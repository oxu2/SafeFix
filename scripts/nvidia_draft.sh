CUDA_VISIBLE_DEVICES=2 nohup python -u predict.py --dataset imagenet --model resnet --num_added_img 1000  --save_pred_model best --attributes pinkcolor_fabrictexture --num_epochs 30  > logs_imagenet/PC_FT_resnet_imagenet__1000.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a redhair_brownskin_sademotion -d 0 -u \
  > logs/batch/batch_resnet_redhair_brownskin_sademotion_nocontrolnet.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 3 -u \
  > logs/batch/batch_resnet_redhair_brownskin_nocontrolnet.log 2>&1 < /dev/null &


nohup ./batch.sh -m resnet -a pinkcolor_fabrictexture -d 0 -s \
  > logs/batch/simple_batch_resnet_pinkcolor_fabrictexture.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a pinkcolor_fabrictexture -d 1 -s \
  > logs/batch/simple_batch_vit_pinkcolor_fabrictexture.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a orangecolor_fabrictexture -d 2 -s \
  > logs/batch/simple_batch_clip_cccc.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a pinkcolor_fabrictexture -d 0 \
  > logs/batch/batch_resnet_pinkcolor_fabrictexture.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a fabrictexture -d 1 \
  > logs/batch/batch_resnet_fabrictexture.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a orangecolor_fabrictexture -d 3 \
  > logs/batch/batch_resnet_orangecolor_fabrictexture.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 python utils/filter.py \
  --attributes pinkcolor_fabrictexture \
  --copy_only  \
  --feasible_json save/feasible_pinkcolor_fabrictexture.json


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --attributes pinkcolor_fabrictexture \
  > logs/filter/filter_pinkcolor_fabrictexture.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --attributes pinkcolor_fabrictexture \
  > logs/filter/filter_pinkcolor_fabrictexture.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u utils/filter.py \
  --attributes fabrictexture \
  > logs/filter/filter_fabrictexture.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u utils/filter.py \
  --attributes yellowcolor \
  > logs/filter/filter_yellowcolor.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u utils/filter.py \
  --attributes yellowcolor \
  > logs/filter/filter_yellowcolor.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup python -u utils/filter.py \
  --attributes pinkcolor \
  > logs/filter/filter_pinkcolor.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --attributes purplecolor \
  > logs/filter/filter_purplecolor.log 2>&1 < /dev/null &



CUDA_VISIBLE_DEVICES=3 nohup python -u utils/filter.py \
  --attributes fabrictexture \
  > logs/filter/filter_fabrictexture.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 python predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 40


CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1 --pretrained 1 > logs/base_imgnet_vit_pretrain_freeze5.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1 --pretrained 1 > logs/base_imgnet_vit_0505pretrain.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1 --pretrained 1 > logs/base_imgnet_vit_pretrain.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model vit --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1  > logs/base_imgnet_vit.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup  python -u predict.py --dataset imagenet --model resnet --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1  > logs/base_imgnet_resnet.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup  python -u predict.py --dataset imagenet --model clip --num_added_img 0  --save_pred_model last --num_epochs 30 --plot 1  > logs/base_imgnet_clip.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 33 \
  --attributes redhair_sademotion \
  > logs/filter_redhair_sademotion.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u controlnet.py --attributes yellowhair_sademotion > logs/control_yellowhair_sademotion.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u controlnet.py > logs/control_redhair_brownskin_sademotion.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 python controlnet.py




nohup ./batch.sh -m clip -a brownskin -d 0 \
  > logs/batch_clip_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a brownskin -d 0 \
  > logs/batch_resnet_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a brownskin -d 1 \
  > logs/batch_vit_brownskin.log 2>&1 < /dev/null &


nohup ./batch.sh -m clip -a sademotion -d 2 \
  > logs/batch_clip_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a sademotion -d 0 \
  > logs/batch_resnet_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a sademotion -d 3 \
  > logs/batch_vit_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a sademotion -d 3 \
  > logs/batch_resnet_sademotion.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair -d 2 -g \
  > logs/g_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair_brownskin -d 0 -g \
  > logs/g_clip_redhair_brownskin.log 2>&1 < /dev/null &

python utils/g_log_compare.py    \
  --prefix g_simple \
  --model clip \
  --attributes redhair

python utils/g_log_compare.py    \
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


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 33 \
  --attributes sademotion \
  > logs/filter_sademotion.log 2>&1 < /dev/null &

❯ CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 33 \
  --attributes brownskin \
  > logs/filter_brownskin2.log 2>&1 < /dev/null &
[1] 2528585
❯ CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 33 \
  --attributes sademotion \
  > logs/filter_sademotion.log 2>&1 < /dev/null &
[2] 2529199


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 33 \
  --attributes yellowhair \
  > logs/filter_yellowhair.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 33 \
  --attributes brownskin \
  > logs/filter_brownskin2.log 2>&1 < /dev/null &

python compare_logs.py

nohup ./batch.sh -m vit -a redhair_brownskin -d 1 \
  > logs/batch_vit_redhair_brownskin_best.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair_brownskin -d 0 \
  > logs/batch_clip_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 1 \
  > logs/batch_clip_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a redhair_brownskin -d 3 \
  > logs/batch_vit_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair -d 2 \
  > logs/batch_vit_redhair.log 2>&1 < /dev/null &

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
  > logs/batch_resnet_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair_brownskin -d 3 \
  > logs/batch_resnet_redhair_brownskin.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 python predict.py --model resnet --num_added_img 1000 --attributes redhair 

CUDA_VISIBLE_DEVICES=2 python -u predict.py --model resnet --num_added_img 1000 --attributes redhair  > logs/test.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=0 nohup python -u utils/filter.py \
  --dir data/sad \
  --attributes sademotion \
  > logs/filter_sad.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0  python utils/filter.py --max_memory 23 --dir data/redhair --attributes redhair
CUDA_VISIBLE_DEVICES=0  python utils/filter.py --max_memory 23 --dir data/sad --attributes sademotion

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

CUDA_VISIBLE_DEVICES=2 nohup python -u utils/filter.py \
  --max_memory 23 \
  --dir hibug_redhair_brownskin \
  --attributes redhair_brownskin \
  --stop_at 15000 \
  > logs/filter_hibug_redhair_brownskin_15000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py \
  --max_memory 23 \
  --dir hibug_redhair \
  --attributes redhair \
  --stop_at 15000 \
  > logs/filter_hibug_redhair_12340.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh sd15.py --attributes brownskin --num_added_img 12340 > logs/sd_brownskin_12340.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 1 > logs/batch_redhair_brownskin_100000.log 2>&1 < /dev/null &

nohup ./for.sh -n 10000 -a redhair -d 3 > logs/for_redhair_10000.log 2>&1 &

nohup ./for.sh -n 10000 -a redhair_brownskin -d 1 -g > logs/g_redhair_brownskin_10000.log 2>&1 &
nohup ./for.sh -n 10000 -a redhair_brownskin -d 2 > logs/for_redhair_brownskin_10000.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u utils/filter.py \
  --dir data/redhair_brownskin \
  --attributes redhair_brownskin \
  --max_memory 23 \
  --feasible_only \
  --results_file save/json/redhair_brownskin_results_04260045.json \
  > logs/filter_redhairb_feasible.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u utils/filter.py \
  --dir data/redhair \
  --attributes redhair \
  --max_memory 23 \
  --feasible_only \
  --results_file save/json/redhair_results_04260005.json \
  > logs/filter_redhair_feasible.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python -u utils/filter.py \
  --max_memory 23 \
  --dir data/redhair_brownskin \
  --attributes redhair_brownskin \
  --stop_at 15000 \
  > logs/filter_redhair_brownskin_15000.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=2 nohup python -u utils/filter.py \
  --max_memory 23 \
  --dir data/redhair_brownskin \
  --attributes redhair_brownskin \
  --stop_at 21111 \
  > logs/filter_redhair_brownskin_21111.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup python -u utils/filter.py \
  --max_memory 23 \
  --dir data/redhair \
  --attributes redhair \
  --stop_at 20000 \
  > logs/filter_redhair_20000.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 3 > logs/batch_clip_redhair.log 2>&1 < /dev/null &

nohup ./batch.sh -m vit -a redhair_brownskin -d 3 > logs/batch_vit_redhair_brownskin.log 2>&1 < /dev/null &
nohup ./for.sh -n 10000 -a redhair_brownskin -d 0 > logs/for_redhair_brownskin_10000.log 2>&1 &
nohup ./for.sh -n 10000 -a redhair_brownskin -d 1 -g > logs/g_redhair_brownskin_10000.log 2>&1 &
nohup ./batch.sh -m clip -a redhair_brownskin -d 2 -g > logs/g_batch_vit_redhair_brownskin.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet --num_added_img 12340 > logs/resnet_12340.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup python -u utils/filter.py --max_memory 23 --dir data/redhair --attributes redhair > logs/filter_redhair_20000.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model clip --num_added_img 0 --plot 1 > logs/clip_0imgs.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet --num_added_img 0 > logs/resnet_0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py --model resnet --num_added_img 0 > logs/resnet_0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model vit --num_added_img 0 > logs/vit_0.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=2,3 python utils/filter.py

nohup ./for.sh -n 0 -a redhair_brownskin -d 3 > logs/for_redhair_brownskin_0.log 2>&1 &
nohup ./for.sh -n 0 -a redhair -d 2 > logs/for_redhair_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,2,3 nohup python -u utils/filter.py --max_memory 11GiB --dir data/redhair_brownskin_46703 > logs/filter_redhair_brownskin_46703.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair -d 1 > logs/batch_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair -d 1 > logs/batch_vit_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair -d 1 -g > logs/batch_resnet_redhair_g.log 2>&1 < /dev/null &

nohup ./batch.sh -m resnet -a redhair_brownskin -d 2 -g > logs/resnet_redhair_brownskin_g.log 2>&1 < /dev/null &

nohup ./batch.sh -m clip -a redhair -d 2 > logs/batch_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m clip -a redhair -d 1 > logs/batch_clip_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m vit -a redhair -d 1 > logs/batch_vit_redhair.log 2>&1 < /dev/null &
nohup ./batch.sh -m resnet -a redhair -d 1 > logs/batch__redhair.log 2>&1 < /dev/null &

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
nohup bash time_run.sh utils/stat.py \
  --predictions predictions_vit_origin_0421030925.npy \
  > logs/stat_vit_origin_0421030925.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 \
nohup bash time_run.sh utils/stat.py \
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
nohup bash time_run.sh utils/stat.py \
  --predictions predictions_clip_origin_0421030416.npy \
  > logs/stat_clip_origin_0421030416.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 \
nohup bash time_run.sh utils/stat.py \
  --predictions pred/predictions_vit_origin_0421030925.npy \
  > logs/compare/predictions_vit_origin_0421030925.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 \
nohup bash time_run.sh utils/stat.py \
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

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh utils/stat.py > logs/stat_vit_no_data.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet18 > logs/resnet18_redhair_15.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup bash time_run.sh predict.py --feasible-file a.txt > logs/resnet_origin_15.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py --feasible-file a.txt > logs/vit_origin_15.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u predict.py --feasible-file a.txt > logs/vit_origin.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup python -u predict.py  > logs/vit_redhair2.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup bash time_run.sh predict.py --model resnet18 --feasible-file a.txt > logs/resnet18_no_pretrained.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=0 nohup bash time_run.sh predict.py --model resnet18 > logs/resnet18_no_data_added.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh predict.py > logs/resnet18_no_data.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh utils/stat.py > logs/red.log 2>&1 < /dev/null &


CUDA_VISIBLE_DEVICES=3 nohup bash time_run.sh utils/stat.py > logs/red.log 2>&1 < /dev/null &



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


CUDA_VISIBLE_DEVICES=3 python utils/stat.py

CUDA_VISIBLE_DEVICES=0 python utils/filter.py

CUDA_VISIBLE_DEVICES=0,2,3 python utils/filter.py


CUDA_VISIBLE_DEVICES=1 nohup python -u utils/filter.py  > logs/filter.log 2>&1 < /dev/null &

CUDA_VISIBLE_DEVICES=1 nohup python -u map.py  > logs/map.log 2>&1 < /dev/null &

# CONDA_ENV=myenv python utils/filter.py --dir /path/to/images --file-type jpg
CONDA_ENV=control python utils/filter.py

CUDA_VISIBLE_DEVICES=1,2,3 python utils/filter.py

CUDA_VISIBLE_DEVICES=1,2,3 python utils/filter.py --verbose True