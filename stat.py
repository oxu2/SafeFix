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
predictions = np.load("combined_predictions_04080238_resnet18.npy")
# predictions = (lambda p: np.concatenate([p[:100000], p[-80000:]], axis=0))(np.load("/people/cs/o/oxx220000/HiBug/combined_predictions_04080238.npy"))
predictions_old = np.load("exampleData/celebA/predictions.npy")[idxs]
print("predictions shape:", predictions.shape)
print("predictions_old shape:", predictions_old.shape)
all_datas = [all_datas[i] for i in idxs]
split = [TRAIN for i in  train_idx] + [VALID for i in  valid_idx] + [UNLABELED for i in  unlabel_idx]


"""
Diagnose the model based on 1. attributes, 2. label, 3. predictions
"""
# # Diagnose
from processData.Diagnose import ModelDiagnose

# Define the parameters
acc_diff_threshold = 0.01
distribution_diff_threshold = 0.3
dominant_label_in_failure_threshold = 0.8
dominant_label_in_prediction_threshold = 0.8
rare_label_threshold = 0.93
dominant_error_threshold = 0.3
error_cover_threshold = 0.9
top_error = 3

MD = ModelDiagnose(labels, 
                   predictions, 
                   split, "exampleData/celebA/attribute.json",
                   acc_diff_threshold=acc_diff_threshold,
                   distribution_diff_threshold=distribution_diff_threshold,
                   dominant_label_in_failure_threshold=dominant_label_in_failure_threshold,
                   dominant_label_in_prediction_threshold=dominant_label_in_prediction_threshold,
                   rare_label_threshold=rare_label_threshold,
                   dominant_error_threshold=dominant_error_threshold,
                   error_cover_threshold=error_cover_threshold,
                   top_error=top_error)

print("Start detect_failure_by_label\n")
MD.detect_failure_by_label()
print("Start detect_prediction_correlation\n")
MD.detect_prediction_correlation()
print("Start detect_failure_prediction_correlation\n")
MD.detect_failure_prediction_correlation()