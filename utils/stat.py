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

import argparse
import numpy as np
from utils.defines import *
import os
# Add necessary imports for ModelDiagnose
import json
from itertools import combinations
from processData.report import ReportGenerator # Assuming ReportGenerator is needed
from utils.defines import *

# Dummy report to disable ReportGenerator on ImageNet
class DummyReport:
    def __getattr__(self, name):
        # All report methods become no-ops
        def method(*args, **kwargs):
            return None
        return method

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', default = None, type=str, required=True, help='Path to predictions .npy file')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Enable verbose output')
    parser.add_argument(
        '--dataset',
        choices=['celeba', 'imagenet'],
        default='celeba',
        help='Which dataset to process: celeba (default) or imagenet'
    )
    args = parser.parse_args()

    dataset = args.dataset
    attribute_file = "" # Initialize attribute file path

    if dataset == 'celeba':
        f = open("exampleData/celebA/list_attr_celeba.txt")
        lines = list(f.readlines())
        lines = [line.strip('\n').split(' ')[0] for line in lines]
        all_datas = ['/data/user/img_align_celeba/' + line for line in lines][1:]
        train_idx = np.load("exampleData/celebA/train_idx.npy")[:80000]
        valid_idx = np.arange(len(all_datas))[-100000:-80000]
        unlabel_idx = np.arange(len(all_datas))[-80000:]
# print(len(all_datas), len(train_idx), len(valid_idx), len(unlabel_idx))
        # that means, the 80001 th to 102599 th sample are not used in this example
# print(len(all_datas), len(train_idx), len(valid_idx), len(unlabel_idx))
        # that means, the 80001 th to 102599 th sample are not used in this example
        idxs = np.concatenate([train_idx, valid_idx, unlabel_idx], axis=0)
        labels = np.load("exampleData/celebA/labels.npy")[idxs]
        predictions = np.load(os.path.join("./save", args.predictions))
        # predictions = (lambda p: np.concatenate([p[:100000], p[-80000:]], axis=0))(np.load("./combined_predictions_04080238.npy"))
        # predictions = np.load("exampleData/celebA/predictions.npy")[idxs]
        # print("predictions shape:", predictions.shape)
        # print("predictions_old shape:", predictions_old.shape)
        all_datas = [all_datas[i] for i in idxs]
        split = [TRAIN for i in  train_idx] + [VALID for i in  valid_idx] + [UNLABELED for i in  unlabel_idx]
        attribute_file = "exampleData/celebA/attribute.json"
    elif dataset == 'imagenet':
        print("Loading ImageNet data...")
        # Load pre-split ImageNet indices
        train_idx = np.load('train_idx_imagenet.npy')
        valid_idx = np.load('val_idx_imagenet.npy')
        # ImageNet doesn't have a predefined unlabeled set in this context
        unlabel_idx = np.array([], dtype=int)

        # Load provided labels for ImageNet subset
        all_labels_full = np.load('labels10.npy')
        # Concatenate labels for train and validation sets based on provided labels
        labels = np.concatenate([all_labels_full[train_idx], all_labels_full[valid_idx]])
        # Build mapping from numeric labels to folder names
        import os as _os  # alias to avoid shadowing
        folder_path = 'imagenet_subset'
        class_names = sorted([d for d in _os.listdir(folder_path) if _os.path.isdir(_os.path.join(folder_path, d))])
        label_names_dict = {i: class_names[i] for i in range(len(class_names))}
        print(f"Loaded and processed {len(labels)} ImageNet labels from labels10.npy.")
        print(f"Class names: {class_names}")

        # Load predictions
        try:
            # Predictions should correspond to the concatenated train+valid indices order
            predictions = np.load(os.path.join("./save", args.predictions))
            # Ensure predictions match the number of labels
            if len(predictions) != len(labels):
                 print(f"Warning: Number of predictions ({len(predictions)}) does not match number of labels ({len(labels)}). Ensure the prediction file corresponds to the train+validation set.")
                 # Attempt to slice if predictions seem to cover more data (e.g., including unlabeled)
                 if len(predictions) > len(labels):
                     print(f"Attempting to use the first {len(labels)} predictions.")
                     predictions = predictions[:len(labels)]
                 else:
                     print("Cannot proceed with mismatched prediction/label counts.")
                     exit()

            print(f"Loaded {len(predictions)} predictions from {args.predictions}")
        except FileNotFoundError:
            print(f"Error: Prediction file {args.predictions} not found in ./save/")
            exit()
        except Exception as e:
            print(f"Error loading predictions: {e}")
            exit()


        # Create split list corresponding to the order of 'labels' and 'predictions'
        split = [TRAIN] * len(train_idx) + [VALID] * len(valid_idx)
        print(f"Created split list: {len(split)} entries ({split.count(TRAIN)} TRAIN, {split.count(VALID)} VALID)")

        # Define attribute file for ImageNet
        attribute_file = "attribute10.json"
        if not os.path.exists(attribute_file):
             print(f"Warning: ImageNet attribute file '{attribute_file}' not found. Run blip.py with --with_attribute_selection.")
             # Decide if you want to exit or continue without attributes
             # exit()
             attribute_file = {} # Use empty dict if file not found

    else:
        print(f"Error: Unknown dataset '{dataset}'")
        exit()

    # Define the parameters
    acc_diff_threshold = 0.01
    distribution_diff_threshold = 0.3
    dominant_label_in_failure_threshold = 0.8
    dominant_label_in_prediction_threshold = 0.8
    rare_label_threshold = 0.93
    dominant_error_threshold = 0.3
    error_cover_threshold = 0.9
    top_error = 3

    # Pass label_names_dict for imagenet, otherwise default empty dict
    if dataset == 'imagenet':
        # Convert any string labels to numeric indices using label_names_dict
        if args.dataset == 'imagenet':
            str2idx = {v: k for k, v in label_names_dict.items()}
            labels = np.array([
                str2idx[lbl] if isinstance(lbl, str) else int(lbl)
                for lbl in labels
            ], dtype=int)
        MD = ModelDiagnose(labels,
                           predictions,
                           split,
                           attribute_file,
                           dataset=dataset,
                           label_names_dict=label_names_dict,
                           print=args.verbose,
                           acc_diff_threshold=acc_diff_threshold,
                           distribution_diff_threshold=distribution_diff_threshold,
                           dominant_label_in_failure_threshold=dominant_label_in_failure_threshold,
                           dominant_label_in_prediction_threshold=dominant_label_in_prediction_threshold,
                           rare_label_threshold=rare_label_threshold,
                           dominant_error_threshold=dominant_error_threshold,
                           error_cover_threshold=error_cover_threshold,
                           top_error=top_error)
    else:
        MD = ModelDiagnose(labels,
                           predictions,
                           split, attribute_file,
                           dataset=dataset,
                           print=args.verbose,
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
    # print("Start detect_failure_prediction_correlation\n")
    # MD.detect_failure_prediction_correlation()


"""
Diagnose the model based on 1. attributes, 2. label, 3. predictions
"""
# # Diagnose
# Remove the import: from processData.Diagnose import ModelDiagnose

# Insert the ModelDiagnose class definition here
class ModelDiagnose():
    def __init__(self, labels, predictions, split, attribute_dict_path, dataset='celeba', label_names_dict={}, print=True,
                 acc_diff_threshold=0.03, distribution_diff_threshold=0.3, dominant_label_in_failure_threshold=0.8,
                 dominant_label_in_prediction_threshold=0.8, rare_label_threshold=0.5, dominant_error_threshold=0.3,
                 error_cover_threshold=0.9, top_error=3) -> None:
        self.labels = labels
        self.dataset = dataset
        self.predictions = predictions
        self._print = print
        self.label_names_dict = label_names_dict if label_names_dict != {} else {label: str(label) for label in set(labels)}
        if type(attribute_dict_path) == str:
            self.attributes = json.load(open(attribute_dict_path))
        else:
            self.attributes = attribute_dict_path
        # Determine numeric label indices, ignoring non-numeric keys
        self.label_number = []
        for key in self.label_names_dict.keys():
            if isinstance(key, int):
                self.label_number.append(key)
            else:
                try:
                    # convert numeric strings to int, skip others
                    num = int(key)
                    self.label_number.append(num)
                except ValueError:
                    # skip non-numeric keys like class names
                    continue
        self.report = {}
        self.split = split
        self.data_length = len(self.labels)
        self.init_data_static()

        # Initialize thresholds
        self.acc_diff_threshold = acc_diff_threshold
        self.distribution_diff_threshold = distribution_diff_threshold
        self.dominant_label_in_failure_threshold = dominant_label_in_failure_threshold
        self.dominant_label_in_prediction_threshold = dominant_label_in_prediction_threshold
        self.rare_label_threshold = rare_label_threshold
        self.dominant_error_threshold = dominant_error_threshold
        self.error_cover_threshold = error_cover_threshold
        self.top_error = top_error

    def init_data_static(self):
        # Get train/valid indexs
        self.train_idx = []
        self.valid_idx = []
        for i in range(self.data_length):
            if self.split[i] == TRAIN:
                self.train_idx.append(i)
            elif self.split[i] == VALID:
                self.valid_idx.append(i)
        self.train_confusion_matrix = np.zeros([len(self.label_number),len(self.label_number)])
        self.valid_confusion_matrix = np.zeros([len(self.label_number),len(self.label_number)])
        self.train_labels_distribution = np.zeros(len(self.label_number))
        self.valid_labels_distribution = np.zeros(len(self.label_number))

        # Train distribution
        for label, prediction in zip(self.labels[self.train_idx], self.predictions[self.train_idx]):
            l = int(label)
            p = int(prediction)
            self.train_confusion_matrix[l][p] += 1
            self.train_labels_distribution[l] += 1

        # Valid distribution
        for label, prediction in zip(self.labels[self.valid_idx], self.predictions[self.valid_idx]):
            l = int(label)
            p = int(prediction)
            self.valid_confusion_matrix[l][p] += 1
            self.valid_labels_distribution[l] += 1

        # Error list
        self.error_binary_list = np.zeros(self.data_length)
        for i, (label, prediction) in enumerate(zip(self.labels, self.predictions)):
            if label != prediction:
                self.error_binary_list[i] = 1

        self.print_output("Model Validation ACC:",1-np.sum(self.error_binary_list[self.valid_idx])/len(self.valid_idx))

        # Attribute distribution
        for attribute_name in self.attributes.keys():
            self.attributes[attribute_name]['distribution'] = {TRAIN:{description:0 for description in self.attributes[attribute_name]['word']},VALID:{description:0 for description in self.attributes[attribute_name]['word']}}
            self.attributes[attribute_name]['valid acc'] = {description:0 for description in self.attributes[attribute_name]['word']}

            for i, description in enumerate(self.attributes[attribute_name]['data']):
                if self.split[i] == TRAIN:
                    self.attributes[attribute_name]['distribution'][TRAIN][description] += 1
                elif self.split[i] == VALID:
                    self.attributes[attribute_name]['distribution'][VALID][description] += 1
                    if self.predictions[i] == self.labels[i]:
                        self.attributes[attribute_name]['valid acc'][description] += 1


            self.attributes[attribute_name]['valid acc'] = {description:self.attributes[attribute_name]['valid acc'][description]/self.attributes[attribute_name]['distribution'][VALID][description] if self.attributes[attribute_name]['distribution'][VALID][description]!=0 else 0 for description in self.attributes[attribute_name]['word'] }

        # Valid acc
        correct = 0
        for label in self.label_number:
            correct += self.train_confusion_matrix[label][label]
        self.train_acc = correct/len(self.valid_idx) if len(self.valid_idx) > 0 else 0 # Avoid division by zero

        correct = 0
        for label in self.label_number:
            correct += self.valid_confusion_matrix[label][label]
        self.valid_acc = correct/len(self.valid_idx) if len(self.valid_idx) > 0 else 0 # Avoid division by zero
        # Use DummyReport for ImageNet to skip report logic
        if self.dataset == 'imagenet':
            self.report = DummyReport()
        else:
            self.report = ReportGenerator(self.attributes, self.train_labels_distribution, self.valid_labels_distribution, self.train_confusion_matrix, self.valid_confusion_matrix, self.valid_acc, self.label_names_dict)

    def generate_report(self, dir="./"):
        self.report.save_report(dir)

    def print_output(self,*args):
        # if self._print:
        if 1:
            print(*args)

    def detect_failure_by_label(self):
        for label in self.label_number:
            if self.valid_labels_distribution[label] > 0:
                class_acc = self.valid_confusion_matrix[label][label]/self.valid_labels_distribution[label]
                if class_acc < self.valid_acc - self.acc_diff_threshold:
                    # Check if train_idx is not empty before division
                    train_label_count = self.train_labels_distribution[label]
                    train_total = len(self.train_idx)
                    if train_total > 0 and train_label_count < self.rare_label_threshold * (train_total/len(self.label_number)):
                        self.report.record_rare_label_error(label)
                        self.print_output('Rare Class: ',label)
                    elif train_total == 0: # Handle case where train set is empty
                         self.report.record_rare_label_error(label) # Or handle differently
                         self.print_output('Rare Class (no training data): ',label)


    def detect_failure_by_sub_label(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'label':
                # Ensure attribute index exists in valid_labels_distribution
                if attribute < len(self.valid_labels_distribution) and self.valid_labels_distribution[attribute] > 0:
                    class_acc = self.valid_confusion_matrix[attribute][attribute]/self.valid_labels_distribution[attribute]
                    for sub_label in attribute_info['word']:
                        sub_acc = self.attributes[attribute]['valid acc'][sub_label]
                        # Ensure attribute index exists in train_labels_distribution
                        if attribute < len(self.train_labels_distribution) and sub_acc < class_acc - self.acc_diff_threshold and self.attributes[attribute]['distribution'][VALID][sub_label] != 0:
                            train_sub_label_count = self.attributes[attribute]['distribution'][TRAIN][sub_label]
                            train_label_total = self.train_labels_distribution[attribute]
                            if train_label_total > 0 and train_sub_label_count < self.rare_label_threshold * (train_label_total/len(attribute_info['word'])):
                                self.report.record_rare_case_error(attribute, sub_label, is_label=True)
                                self.print_output('Rare Sub Class: ',sub_label, attribute)
                            elif train_label_total == 0: # Handle case with no training data for the main label
                                self.report.record_rare_case_error(attribute, sub_label, is_label=True)
                                self.print_output('Rare Sub Class (no training data for label %s): '%attribute,sub_label, attribute)


    def detect_prediction_correlation(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'description' or attribute_info['type'] == 'binary':
                for description in attribute_info['word']:
                    sub_acc = self.attributes[attribute]['valid acc'][description]

                    

                    if attribute == "hair color" and description in ["white", "yellow", "blonde", "black", "gray", "brown", "red"]:
                        description_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == description]
                        if self._print:
                            self.print_output(f"Indices with {attribute}={description} in validation set: {description_valid_idx[20, 30, 40]}...")
                        
                        self.print_output(f'The accuracy with "{attribute}" = "{description}" is {sub_acc}')
                        correct_count = sum(1 for idx in description_valid_idx if self.predictions[idx] == self.labels[idx])
                        self.print_output(f"Correct predictions: {correct_count}/{len(description_valid_idx)}")

                    if attribute == "skin" and description in ["white", "black", "brown"]:
                        description_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == description]
                        if self._print:
                            self.print_output(f"Indices with {attribute}={description} in validation set: {description_valid_idx[:3]}...")
                        self.print_output(f'The accuracy with "{attribute}" = "{description}" is {sub_acc}')
                        correct_count = sum(1 for idx in description_valid_idx if self.predictions[idx] == self.labels[idx])
                        self.print_output(f"Correct predictions: {correct_count}/{len(description_valid_idx)}")
                        
                    if sub_acc < self.valid_acc - self.acc_diff_threshold:
                        if len(self.train_idx):
                            valid_distribution = self.attributes[attribute]['distribution'][VALID][description] / len(self.valid_idx) if len(self.valid_idx) else 0
                            train_distribution = self.attributes[attribute]['distribution'][TRAIN][description] / len(self.train_idx) if len(self.train_idx) else 0
                            if valid_distribution - train_distribution > self.distribution_diff_threshold:
                                self.report.record_rare_case_error(attribute, description, distribution_shift=True)
                                self.print_output('ACC: %f \tDistribution Shift in attribute: "%s", description: %s, train: %f, valid: %f"' % (sub_acc, attribute, description, train_distribution, valid_distribution))
                            elif train_distribution < self.rare_label_threshold * (1 / len(self.attributes[attribute]['word'])):
                                self.report.record_rare_case_error(attribute, description, is_rare=True)
                                self.print_output('ACC: %f \tRare Case: attribute "%s", description "%s"' % (sub_acc, attribute, description))

                                # if attribute == "hair color" and description == "red":
                                #     description_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == description]
                                #     if len(description_valid_idx) > 20:
                                #         self.print_output(f"Indices with {attribute}={description} in validation set: {description_valid_idx[20]}...")
                                #     else:
                                #         self.print_output(f"Indices with {attribute}={description} in validation set: {description_valid_idx}...")
                                #     self.print_output(f"The accuracy {sub_acc} is calculated from validation set samples with this attribute")
                                #     correct_count = sum(1 for idx in description_valid_idx if self.predictions[idx] == self.labels[idx])
                                #     self.print_output(f"Correct predictions: {correct_count}/{len(description_valid_idx)}")
                            else:
                                self.report.record_rare_case_error(attribute, description)
                                self.print_output('ACC: %f \tHard case: attribute "%s", description "%s"' % (sub_acc, attribute, description))


    def detect_failure_prediction_correlation(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'description' or attribute_info['type'] == 'binary':
                for description in attribute_info['word']:

                    description_error_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == description and self.error_binary_list[i]]
                    if len(description_error_valid_idx) > 0: # Avoid division by zero
                        distribution = np.zeros(len(self.label_number))

                        for idx in description_error_valid_idx:
                            raw_label = self.predictions[idx]
                            # Cast to integer index, skip if invalid
                            try:
                                label_int = int(raw_label)
                            except (ValueError, TypeError):
                                continue
                            if 0 <= label_int < len(distribution):
                                distribution[label_int] += 1

                        distribution = distribution/len(description_error_valid_idx)
                        largest_label = np.argmax(distribution)
                        # non_largest_label_prob = 1 - distribution[largest_label] # This variable is calculated but not used

                        if distribution[largest_label] > self.dominant_label_in_failure_threshold:
                            self.report.record_correlation_error(attribute, description, label_distribution=distribution, in_failure=True)
                            # ... (existing commented out print statements) ...

                            combined_distribution = np.zeros(len(self.label_number))
                            other_error_indices = []
                            for other_description in attribute_info['word']:
                                if other_description != description:
                                    other_description_error_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == other_description and self.error_binary_list[i]]
                                    other_error_indices.extend(other_description_error_valid_idx)

                            if len(other_error_indices) > 0: # Avoid division by zero if no errors in other descriptions
                                for idx in other_error_indices:
                                    raw_label = self.predictions[idx]
                                    try:
                                        label_int = int(raw_label)
                                    except (ValueError, TypeError):
                                        continue
                                    if 0 <= label_int < len(combined_distribution):
                                        combined_distribution[label_int] += 1

                                combined_distribution = combined_distribution / np.sum(combined_distribution) # Normalization happens here
                                # self.print_output("While attribute %s is not %s, combined prediction is %s with prob %s"%(attribute, description, str(np.argmax(combined_distribution)), str(np.max(combined_distribution))))


    def detect_failure_in_attribute_combination(self, combine_num = 3, label_as_attribute=False, high_cover_combinations=False, label=None):
        attributes = self.attributes.copy() # Use a copy to avoid modifying the original dict if label_as_attribute is True
        if label_as_attribute:
            attributes['LABEL'] = {}
            attributes['LABEL']['data'] = [str(i) for i in self.labels]
            attributes['LABEL']['word'] = [str(i) for i in self.label_number]

        if label is None:
            data_idx = self.valid_idx
        else:
            # Ensure label is valid before filtering
            if label in self.label_number:
                data_idx = [i for i in self.valid_idx if self.labels[i] == label]
            else:
                self.print_output(f"Warning: Label {label} not found in label_number. Using all validation data.")
                data_idx = self.valid_idx

        if not data_idx: # Handle case where data_idx is empty
            self.print_output("Warning: No validation data found for the specified criteria.")
            return

        # Generate combinations of "combine_num" attributes
        combine_num = min(len(attributes.keys()), combine_num)
        all_attribute_combineation_set = []
        # Ensure there are enough attributes to form combinations
        if len(attributes.keys()) >= combine_num:
            combinations_list = list(combinations(list(attributes.keys()),combine_num))

            for a_combination in combinations_list:
                attribute_combineation_set = [[[],np.array(data_idx, dtype=int)],] # Start with all data_idx
                valid_combination = True
                for attribute in a_combination:
                    if attribute not in attributes: # Check if attribute exists
                        valid_combination = False
                        break
                    new_set = []
                    for description in attributes[attribute]['word']:
                        # Ensure data access is safe
                        description_valid_idx = [i for i in data_idx if i < len(attributes[attribute]['data']) and attributes[attribute]['data'][i] == description]
                        description_valid_idx = np.array(description_valid_idx, dtype=int)

                        for (combination,index) in attribute_combineation_set:
                            if index.size > 0 and description_valid_idx.size > 0:
                                new_idx = np.intersect1d(index, description_valid_idx, assume_unique=True)
                                if new_idx.size > 0: # Only add if intersection is not empty
                                     new_set.append([combination+[attribute + ':' + description,], new_idx])
                            elif index.size == 0 and description_valid_idx.size > 0 and not combination: # Handle initial case
                                new_set.append([combination+[attribute + ':' + description,], description_valid_idx])


                    attribute_combineation_set = new_set
                    if not attribute_combineation_set: # If no valid combinations found for this attribute, break early
                        break
                if valid_combination and attribute_combineation_set:
                    all_attribute_combineation_set += attribute_combineation_set

        # print('len', len(all_attribute_combineation_set))
        # No need to filter for non-empty here as it's handled during creation

        # Compute accuracy, error cover, data cover
        error_cover_list = []
        data_cover_list = [] # Renamed to avoid conflict
        acc_list = []
        all_error = np.sum(self.error_binary_list[data_idx])

        valid_combinations_for_report = [] # Store only combinations with data

        for a_set in all_attribute_combineation_set:
            indices = a_set[1]
            num_indices = len(indices)
            if num_indices > 0:
                error_count = np.sum(self.error_binary_list[indices])
                current_error_cover = error_count / all_error if all_error > 0 else 0
                current_data_cover = num_indices / len(data_idx) if len(data_idx) > 0 else 0
                current_acc = 1 - (error_count / num_indices) if num_indices > 0 else 1.0 # Accuracy is 1 if no data/no errors

                error_cover_list.append(current_error_cover)
                data_cover_list.append(current_data_cover)
                acc_list.append(current_acc)
                valid_combinations_for_report.append(a_set) # Add this valid set

        if valid_combinations_for_report: # Check if there are valid combinations before recording
            self.report.record_combinations(valid_combinations_for_report, error_cover_list, data_cover_list, acc_list)
        else:
             self.print_output("No valid attribute combinations found to record.")


        # Greedy algorithm to find combinations to cover self.error_cover_threshold(90% default) errors
        if high_cover_combinations and valid_combinations_for_report and all_error > 0:
            combination_result, index_result, error_cover_result, data_cover_result, acc_result = [], np.array([], dtype=int), [], [], []
            cover_rate = 0
            # Use a copy of valid combinations to modify during iteration
            remaining_combinations = [(c[0], c[1], ec, dc, ac) for c, ec, dc, ac in zip(valid_combinations_for_report, error_cover_list, data_cover_list, acc_list)]

            while cover_rate < self.error_cover_threshold and remaining_combinations:
                best_gain = -1
                best_idx = -1
                best_new_index = np.array([], dtype=int)
                current_error_count = np.sum(self.error_binary_list[index_result])

                for i, (comb, indices, _, _, _) in enumerate(remaining_combinations):
                    # Calculate gain only from new indices added
                    new_indices_to_add = np.setdiff1d(indices, index_result, assume_unique=True)
                    gain = np.sum(self.error_binary_list[new_indices_to_add])

                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_new_index = np.union1d(index_result, indices) # Calculate the potential new index set

                if best_idx != -1 and best_gain > 0: # Ensure we are adding combinations that increase error coverage
                    best_combination_data = remaining_combinations.pop(best_idx)
                    index_result = best_new_index
                    combination_result.append(best_combination_data[0]) # Append combination description
                    new_total_error_count = np.sum(self.error_binary_list[index_result])
                    cover_rate = new_total_error_count / all_error
                    error_cover_result.append(cover_rate)
                    current_data_cover = len(index_result) / len(data_idx) if len(data_idx) > 0 else 0
                    data_cover_result.append(current_data_cover)
                    current_acc = 1 - (new_total_error_count / len(index_result)) if len(index_result) > 0 else 1.0
                    acc_result.append(current_acc)
                else:
                    # No combination adds more error coverage, or no combinations left
                    break

            if combination_result: # Check if any combinations were selected
                self.print_output("High Error Coverage Attribute Combinations: ",*combination_result)
                self.report.record_high_coverage_combinations(combination_result, error_cover_result, data_cover_result, acc_result)
            else:
                self.print_output("Could not find high error coverage combinations meeting criteria.")
        elif high_cover_combinations:
             self.print_output("Skipping high coverage combination search: No valid combinations or no errors in the dataset.")


    def unlabel_selection(self, error_count=5, cover_threshold=0.9):
        # Ensure report has suggestions before proceeding
        if not hasattr(self.report, 'suggestion_for_combinations_by_rate'):
            self.print_output("Report suggestions not generated. Run combination analysis first.")
            return []
        suggestions = self.report.suggestion_for_combinations_by_rate()
        if not suggestions:
            self.print_output("No suggestions available for unlabel selection.")
            return []

        unlabel_idx = np.array([i for i in range(len(self.split)) if self.split[i]==UNLABELED])
        if len(unlabel_idx) == 0:
            self.print_output("No unlabeled data available.")
            return []

        selected = []
        covered_error_estimate = 0 # Use estimate from suggestions
        count = 0
        selected_indices_set = set() # Use a set for faster lookups and uniqueness

        # Calculate total errors in unlabeled set for actual coverage calculation (if needed for printing)
        unlabeled_errors = self.error_binary_list[unlabel_idx]
        total_unlabeled_error_count = np.sum(unlabeled_errors)


        for suggestion in suggestions:
            # Check if suggestion format is valid
            if not isinstance(suggestion, dict) or 'combination' not in suggestion or not suggestion['combination'] or 'error_cover' not in suggestion or 'acc' not in suggestion:
                self.print_output(f"Skipping invalid suggestion: {suggestion}")
                continue

            count += 1
            # Stop if enough suggestions processed OR estimated error cover threshold reached
            if count > error_count and covered_error_estimate >= cover_threshold:
                 break

            combination = suggestion['combination'][0] # Assuming combination is a list containing the actual list of strings
            if not isinstance(combination, list):
                 self.print_output(f"Skipping suggestion with invalid combination format: {combination}")
                 continue

            current_suggestion_indices = set()
            for i in unlabel_idx:
                match = True # Assume match initially
                for attribute_description in combination:
                    try:
                        attribute, description = attribute_description.split(':', 1) # Split only once
                        # Basic safety checks
                        if attribute not in self.attributes or i >= len(self.attributes[attribute]['data']):
                             match = False
                             break
                        if self.attributes[attribute]['data'][i] != description:
                            match = False
                            break
                    except ValueError: # Handle cases where split fails
                        self.print_output(f"Warning: Could not parse attribute-description '{attribute_description}' in combination {combination}. Skipping index {i}.")
                        match = False
                        break
                    except KeyError: # Handle cases where attribute might be missing after split
                         self.print_output(f"Warning: Attribute '{attribute}' not found for index {i}. Skipping.")
                         match = False
                         break

                if match:
                    current_suggestion_indices.add(i)

            # Update selected set and estimated coverage
            newly_added_count = len(current_suggestion_indices - selected_indices_set)
            if newly_added_count > 0:
                 selected_indices_set.update(current_suggestion_indices)
                 # Add the suggestion's error cover estimate. This is an approximation.
                 covered_error_estimate += suggestion['error_cover']


            # Print statistics (optional, consider frequency)
            if len(selected_indices_set) > 0 and newly_added_count > 0: # Print only if something changed
                selected_list_for_stats = list(selected_indices_set)
                actual_error_count_selected = np.sum(self.error_binary_list[selected_list_for_stats])
                actual_acc = 1 - (actual_error_count_selected / len(selected_list_for_stats)) if len(selected_list_for_stats) > 0 else 1.0
                actual_error_cover = actual_error_count_selected / total_unlabeled_error_count if total_unlabeled_error_count > 0 else 0.0

                self.print_output(f"Suggestion {count}/{len(suggestions)}: Added {newly_added_count} indices.")
                self.print_output(f"  Est. Acc: {suggestion['acc']:.4f}, Est. Error Cover (this suggestion): {suggestion['error_cover']:.4f}")
                self.print_output(f"  Total Selected: {len(selected_indices_set)} ({len(selected_indices_set)/len(unlabel_idx):.4f} of unlabeled)")
                self.print_output(f"  Cumulative Est. Error Cover: {covered_error_estimate:.4f}")
                if total_unlabeled_error_count > 0:
                     self.print_output(f"  Actual Acc (selected): {actual_acc:.4f}, Actual Error Cover (selected): {actual_error_cover:.4f}")
                else:
                     self.print_output(f"  Actual Acc (selected): {actual_acc:.4f} (No errors in unlabeled set)")


        return list(selected_indices_set)


    def unlabel_selection_with_confidence(self, confidence, budget, error_count=5, cover_threshold=0.8, distribution_aware=True):
        import tqdm # Keep tqdm import local if only used here

        # --- Basic Input Validation ---
        if not isinstance(confidence, (np.ndarray, list)) or len(confidence) != self.data_length:
             raise ValueError("Confidence scores must be a list or numpy array matching data length.")
        if not isinstance(budget, int) or budget <= 0:
             raise ValueError("Budget must be a positive integer.")

        # --- Get Suggestions ---
        if not hasattr(self.report, 'suggestion_for_combinations_by_rate'):
            self.print_output("Report suggestions not generated. Run combination analysis first.")
            return []
        suggestions = self.report.suggestion_for_combinations_by_rate()
        if not suggestions:
            self.print_output("No suggestions available for unlabel selection.")
            return []

        # --- Identify Unlabeled Indices ---
        unlabel_idx = np.array([i for i in range(len(self.split)) if self.split[i]==UNLABELED])
        if len(unlabel_idx) == 0:
            self.print_output("No unlabeled data available.")
            return []

        # --- Filter Suggestions and Map Indices ---
        suggestions_selected = []
        suggestion_indices_map = {} # Map suggestion (tuple of combo strings) -> set of indices
        all_suggested_indices = set()
        covered_error_estimate = 0
        count = 0

        self.print_output("Processing suggestions to identify relevant unlabeled indices...")
        for suggestion in tqdm.tqdm(suggestions, desc="Filtering Suggestions"):
             # Validate suggestion format
            if not isinstance(suggestion, dict) or 'combination' not in suggestion or not suggestion['combination'] or 'error_cover' not in suggestion or 'acc' not in suggestion:
                self.print_output(f"Skipping invalid suggestion: {suggestion}")
                continue

            count += 1
            if count > error_count and covered_error_estimate >= cover_threshold:
                self.print_output(f"Stopping suggestion processing: Count ({count}) or Cover ({covered_error_estimate:.4f}) limit reached.")
                break

            combination = suggestion['combination'][0] # Assuming list within list
            if not isinstance(combination, list):
                 self.print_output(f"Skipping suggestion with invalid combination format: {combination}")
                 continue
            # Use tuple for dict key
            combination_key = tuple(sorted(combination))


            current_suggestion_indices = set()
            for i in unlabel_idx:
                match = True
                for attribute_description in combination:
                    try:
                        attribute, description = attribute_description.split(':', 1)
                        if attribute not in self.attributes or i >= len(self.attributes[attribute]['data']) or self.attributes[attribute]['data'][i] != description:
                            match = False
                            break
                    except (ValueError, KeyError):
                        match = False
                        break
                if match:
                    current_suggestion_indices.add(i)

            if current_suggestion_indices:
                suggestions_selected.append(suggestion)
                suggestion_indices_map[combination_key] = current_suggestion_indices
                all_suggested_indices.update(current_suggestion_indices)
                covered_error_estimate += suggestion['error_cover'] # Accumulate estimated cover

                # Optional: Print progress during filtering
                # self.print_output(f"Suggestion {count}: Found {len(current_suggestion_indices)} indices. Cumulative Est. Cover: {covered_error_estimate:.4f}")


        if not all_suggested_indices:
            self.print_output("No unlabeled indices match the selected suggestions.")
            return []

        self.print_output(f"Identified {len(all_suggested_indices)} potentially relevant unlabeled indices across {len(suggestions_selected)} suggestions.")

        # --- Sort ALL potentially relevant indices by confidence ---
        # Filter confidence scores and indices to only include relevant unlabeled ones
        relevant_unlabeled_indices = list(all_suggested_indices)
        relevant_confidences = confidence[relevant_unlabeled_indices]
        # Sort relevant indices based on their confidence scores (ascending, low confidence first)
        sorted_relevant_indices = [idx for _, idx in sorted(zip(relevant_confidences, relevant_unlabeled_indices))]


        # --- Select Final Budget ---
        final_selection = []
        if not distribution_aware:
            self.print_output("Selecting based purely on confidence (lowest first)...")
            final_selection = sorted_relevant_indices[:budget]
        else:
            self.print_output("Selecting based on distribution-aware strategy...")
            # Calculate total estimated error cover from selected suggestions for normalization
            total_selected_error_cover = sum(s['error_cover'] for s in suggestions_selected)
            if total_selected_error_cover <= 0:
                 self.print_output("Warning: Total estimated error cover is zero. Falling back to confidence-based selection.")
                 final_selection = sorted_relevant_indices[:budget]
            else:
                selected_indices_set = set() # Track already selected indices
                # Iterate through suggestions, allocating budget proportionally
                # Sort suggestions by error cover (desc) or accuracy (desc) or keep original order? Let's keep original order for now.
                for suggestion in tqdm.tqdm(suggestions_selected, desc="Allocating Budget"):
                    combination_key = tuple(sorted(suggestion['combination'][0]))
                    suggestion_idxs_set = suggestion_indices_map.get(combination_key, set())

                    # Calculate proportional budget for this suggestion
                    proportional_budget = int(np.ceil(budget * (suggestion['error_cover'] / total_selected_error_cover)))
                    if proportional_budget == 0 and budget > 0: # Ensure at least 1 if possible
                        proportional_budget = 1


                    count_for_this_suggestion = 0
                    # Iterate through globally sorted indices
                    for idx in sorted_relevant_indices:
                        # Check if budget is full
                        if len(selected_indices_set) >= budget:
                            break
                        # Check if index belongs to this suggestion AND hasn't been selected yet
                        if idx in suggestion_idxs_set and idx not in selected_indices_set:
                             # Check if this suggestion's proportional budget is met
                            if count_for_this_suggestion < proportional_budget:
                                selected_indices_set.add(idx)
                                count_for_this_suggestion += 1
                            # else: # Budget for this suggestion met, move to next index for other suggestions
                                # continue # Not strictly needed, outer loop continues

                    # Break outer loop if main budget is full
                    if len(selected_indices_set) >= budget:
                         break

                # Fill remaining budget if needed (due to rounding or empty suggestions)
                if len(selected_indices_set) < budget:
                    self.print_output(f"Filling remaining budget ({budget - len(selected_indices_set)} slots) with lowest confidence indices...")
                    remaining_needed = budget - len(selected_indices_set)
                    fill_count = 0
                    for idx in sorted_relevant_indices:
                        if idx not in selected_indices_set:
                            selected_indices_set.add(idx)
                            fill_count += 1
                            if fill_count >= remaining_needed:
                                break

                final_selection = list(selected_indices_set)


        self.print_output(f"Final selection contains {len(final_selection)} indices.")
        # Optional: Calculate and print stats for the final selection
        if final_selection:
            final_errors = self.error_binary_list[final_selection]
            final_acc = 1 - (np.sum(final_errors) / len(final_selection))
            # You might need total unlabeled errors calculated earlier for coverage
            # final_error_cover = np.sum(final_errors) / total_unlabeled_error_count if total_unlabeled_error_count > 0 else 0.0
            self.print_output(f"  Actual Accuracy of final selection: {final_acc:.4f}")
            # self.print_output(f"  Actual Error Cover of final selection: {final_error_cover:.4f}")


        return final_selection[:budget] # Ensure exactly budget size is returned


    def date_generation(self, combine_num=5, attribute_num=200, address="exampleData/Imagenet10/generation.json", label_specific=True):
        # Ensure report has suggestions before proceeding
        if not hasattr(self.report, 'suggestion_for_combinations_by_rate'):
            self.print_output("Report suggestions not generated. Run combination analysis first.")
            return "{}" # Return empty JSON string

        suggestions = self.report.suggestion_for_combinations_by_rate()
        if not suggestions:
            self.print_output("No suggestions available for data generation.")
            return "{}"

        generation_dict = {}

        if label_specific:
             # This part seems incorrect as suggestions are not label-specific by default.
             # The original code assigns the *same* top suggestions to *every* label.
             # Replicating original behavior:
             self.print_output("Warning: Using the same top global suggestions for each label (original behavior).")
             top_suggestions = suggestions[:attribute_num]
             generate_used = []
             for suggestion in top_suggestions:
                 if isinstance(suggestion, dict) and 'combination' in suggestion and suggestion['combination']:
                     combination = suggestion['combination'][0] # list of "attr:desc"
                     if isinstance(combination, list):
                         # Extract only descriptions
                         descriptions = [item.split(':', 1)[1] for item in combination if ':' in item]
                         if descriptions: # Only add if descriptions were extracted
                             generate_used.append(descriptions)

             for label in self.label_number:
                 generation_dict[str(label)] = generate_used # Use string key for JSON
        else:
            # Use top N suggestions globally, not per label
            self.print_output("Generating global attribute combinations (not label-specific).")
            top_suggestions = suggestions[:attribute_num]
            generate_used = []
            for suggestion in top_suggestions:
                 if isinstance(suggestion, dict) and 'combination' in suggestion and suggestion['combination']:
                     combination = suggestion['combination'][0] # list of "attr:desc"
                     if isinstance(combination, list):
                         # Extract only descriptions
                         descriptions = [item.split(':', 1)[1] for item in combination if ':' in item]
                         if descriptions:
                             generate_used.append(descriptions)
            generation_dict["global"] = generate_used # Use a single key

        try:
            with open(address,'w') as f:
                json.dump(generation_dict, f, indent=4)
            self.print_output(f"Generation attributes saved to {address}")
            # Return the JSON string as well
            return json.dumps(generation_dict, indent=4)
        except IOError as e:
            self.print_output(f"Error writing generation file to {address}: {e}")
            return "{}" # Return empty JSON string on error
        except TypeError as e:
             self.print_output(f"Error serializing generation data to JSON: {e}")
             return "{}"



if __name__ == "__main__":
    main()