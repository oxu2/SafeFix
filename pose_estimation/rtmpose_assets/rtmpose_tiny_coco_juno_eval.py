_base_ = 'mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'

data_root = '/home/oxx220000/scratch/safefix_cache/data/coco/'
randomness = dict(seed=42, deterministic=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=_base_.val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    outfile_prefix='log_pose/rtmpose_tiny_coco_val/rtmpose_tiny_coco_val',
)
test_evaluator = val_evaluator
