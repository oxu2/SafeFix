# Load images, predictions, labels
"""
Here is an example use case of our system.
To use this example, ou need to get celebA dataset first. At http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

all_datas: a list of image address
labels: a list of labels
predictions: a list of predictions
split: a list of tags, tag can be {TRAIN, VALID, UNLABELED}
The above four list should have equal length
"""
import numpy as np
from utils.defines import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
f = open("exampleData/celebA/list_attr_celeba.txt")
lines = list(f.readlines())
lines = [line.strip('\n').split(' ')[0] for line in lines]
all_datas = ['/data/oxx220000/img_align_celeba/' + line for line in lines][1:]
train_idx = np.load("exampleData/celebA/train_idx.npy")[:80000]
valid_idx = np.arange(len(all_datas))[-100000:-80000]
unlabel_idx = np.arange(len(all_datas))[-80000:]
# print(len(all_datas), len(train_idx), len(valid_idx), len(unlabel_idx))
# that means, the 80001 th to 102599 th sample are not used in this example
idxs = np.concatenate([train_idx, valid_idx, unlabel_idx], axis=0)
labels = np.load("exampleData/celebA/labels.npy")[idxs]
predictions = np.load("exampleData/celebA/predictions.npy")[idxs]
all_datas = [all_datas[i] for i in idxs]
split = [TRAIN for i in  train_idx] + [VALID for i in  valid_idx] + [UNLABELED for i in  unlabel_idx]


"""
Diagnose the model based on 1. attributes, 2. label, 3. predictions
"""
# # Diagnose
from processData.Diagnose import ModelDiagnose
MD = ModelDiagnose(labels, predictions, split, "exampleData/celebA/attribute.json")
print("Start detect_failure_by_label\n")
MD.detect_failure_by_label()
print("Start detect_failure_by_prediction\n")
MD.detect_prediction_correlation()
print("Start detect_failure_by_attribute\n")
MD.detect_failure_prediction_correlation()