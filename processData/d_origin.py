import numpy as np
from utils.defines import *
from processData.report import ReportGenerator
import json
from itertools import combinations

ACC_DIFF_THRESHOLD = 0.03
DISTRIBUTION_DIFF_THRESHOLD = 0.3
DOMINANT_LABEL_IN_FAILURE_THRESHOLD = 0.8
DOMINANT_LABEL_IN_PREDICTION_THRESHOLD = 0.8
RARE_LABEL_THRESHOLD = 0.5
DOMINANT_ERROR_THRESHOLD = 0.3
ERROR_COVER_THRESHOLD = 0.9
TOP_ERROR = 3

class ModelDiagnose():
    def __init__(self, labels, predictions, split, attribute_dict_path, label_names_dict={}, print = True) -> None:
        self.labels = labels
        self.predictions = predictions
        self._print = print
        self.label_names_dict = label_names_dict if label_names_dict!={} else {label:str(label) for label in set(labels)}
        if type(attribute_dict_path) == str:
            self.attributes = json.load(open(attribute_dict_path))
        else:
            self.attributes = attribute_dict_path
        self.label_number = [int(label) for label in list(self.label_names_dict.keys())]
        self.report = {}
        self.split = split
        self.data_length = len(self.labels)
        self.init_data_static()

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
        for label,prediction in zip(self.labels[self.train_idx],self.predictions[self.train_idx]):
            self.train_confusion_matrix[label][prediction] += 1
            self.train_labels_distribution[label] += 1

        # Valid distribution
        for label,prediction in zip(self.labels[self.valid_idx],self.predictions[self.valid_idx]):
            self.valid_confusion_matrix[label][prediction] += 1
            self.valid_labels_distribution[label] += 1
        
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
        self.train_acc = correct/len(self.valid_idx)

        correct = 0
        for label in self.label_number:
            correct += self.valid_confusion_matrix[label][label]
        self.valid_acc = correct/len(self.valid_idx)
        self.report = ReportGenerator(self.attributes, self.train_labels_distribution, self.valid_labels_distribution, self.train_confusion_matrix, self.valid_confusion_matrix, self.valid_acc, self.label_names_dict)

    def generate_report(self, dir="./"):
        self.report.save_report(dir)

    def print_output(self,*args):
        if self._print:
            print(*args)
            
    def detect_failure_by_label(self):
        for label in self.label_number:
            if self.valid_labels_distribution[label] > 0:
                class_acc = self.valid_confusion_matrix[label][label]/self.valid_labels_distribution[label]
                if class_acc < self.valid_acc - ACC_DIFF_THRESHOLD:
                    if self.train_labels_distribution[label] < RARE_LABEL_THRESHOLD * (len(self.train_idx)/len(self.label_number)):
                        self.report.record_rare_label_error(label)
                        self.print_output('Rare Class: ',label)

    def detect_failure_by_sub_label(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'label':
                if self.valid_labels_distribution[attribute] > 0:
                    class_acc = self.valid_confusion_matrix[attribute][attribute]/self.valid_labels_distribution[attribute]
                for sub_label in attribute_info['word']:
                    sub_acc = self.attributes[attribute]['valid acc'][sub_label]
                    if sub_acc < class_acc - ACC_DIFF_THRESHOLD and self.attributes[attribute]['distribution'][VALID][sub_label] != 0:
                        if self.attributes[attribute]['distribution'][TRAIN][sub_label] < RARE_LABEL_THRESHOLD * (len(self.train_labels_distribution[attribute])/len(attribute_info['word'])):
                            self.report.record_rare_case_error(attribute, sub_label, is_label=True)
                            self.print_output('Rare Sub Class: ',sub_label, attribute)

    def detect_prediction_correlation(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'description' or attribute_info['type'] == 'binary':
                for description in attribute_info['word']:
                    sub_acc = self.attributes[attribute]['valid acc'][description]
                    if sub_acc < self.valid_acc - ACC_DIFF_THRESHOLD:
                        if len(self.train_idx):
                            valid_distribution = self.attributes[attribute]['distribution'][VALID][description] / len(self.valid_idx) if len(self.valid_idx) else 0
                            train_distribution = self.attributes[attribute]['distribution'][TRAIN][description] / len(self.train_idx) if len(self.train_idx) else 0
                            if valid_distribution - train_distribution > DISTRIBUTION_DIFF_THRESHOLD:
                                self.report.record_rare_case_error(attribute, description, distribution_shift=True)
                                self.print_output('ACC: %f \tDistribution Shift in attribute: "%s", description: %s, train: %f, valid: %f"'%(sub_acc,attribute, description,train_distribution,valid_distribution))
                            elif train_distribution < RARE_LABEL_THRESHOLD * (1/len(self.attributes[attribute]['word'])):
                                self.report.record_rare_case_error(attribute, description, is_rare=True)
                                self.print_output('ACC: %f \tRare Case: attribute "%s", description %s'%(sub_acc,attribute, description))
                            else:
                                self.report.record_rare_case_error(attribute, description)
                                self.print_output('ACC: %f \tHard case: attribute "%s", description %s'%(sub_acc,attribute, description))
                        
                        description_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == description]
                        distribution = np.zeros(len(self.label_number))

                        for idx in description_valid_idx:
                            label = self.predictions[idx]
                            distribution[label] += 1
                        distribution = distribution/len(description_valid_idx)
                        largest_label = np.argmax(distribution)
                        
                        if distribution[largest_label] > DOMINANT_LABEL_IN_PREDICTION_THRESHOLD:
                            self.report.record_correlation_error(attribute, description,label_distribution=distribution)
                            self.print_output("Suspicous Correlation in prediction, when attribute %s is %s, prediction is %s with prob %s"%(attribute, description, str(largest_label), str(distribution[largest_label])))
                            
    def detect_failure_prediction_correlation(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'description' or attribute_info['type'] == 'binary':
                for description in attribute_info['word']:

                    description_error_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == description and self.error_binary_list[i]]
                    distribution = np.zeros(len(self.label_number))

                    for idx in description_error_valid_idx:
                        if self.error_binary_list[idx]:
                            label = self.predictions[idx]
                            distribution[label] += 1
                    
                    distribution = distribution/len(description_error_valid_idx)
                    largest_label = np.argmax(distribution)

                    if distribution[largest_label] > DOMINANT_LABEL_IN_FAILURE_THRESHOLD:
                        self.report.record_correlation_error(attribute, description, label_distribution=distribution, in_failure=True)
                        self.print_output("Suspicous Correlation in failure prediction, when attribute %s is %s, prediction is %s with prob %s"%(attribute, description, str(largest_label), str(distribution[largest_label])))

    def detect_failure_in_attribute_combination(self, combine_num = 3, label_as_attribute=False, high_cover_combinations=False, label=None):
        attributes = self.attributes
        if label_as_attribute:
            attributes['LABEL'] = {}
            attributes['LABEL']['data'] = [str(i) for i in self.labels]
            attributes['LABEL']['word'] = [str(i) for i in self.label_number]
            
        if label is None:
            data_idx = self.valid_idx
        else:
            data_idx = [i for i in self.valid_idx if self.labels[i] == label]

        # Generate combinations of "combine_num" attributes
        combine_num = min(len(attributes.keys()), combine_num)
        all_attribute_combineation_set = []
        combinations_list = [i for i in list(combinations(list(attributes.keys()),combine_num))]
        
        for a_combination in combinations_list:
            attribute_combineation_set = [[[],[]],]
            for attribute in a_combination:
                new_set = []
                for description in attributes[attribute]['word']:
                    description_valid_idx = [i for i in data_idx if attributes[attribute]['data'][i] == description]
                    for (combination,index) in attribute_combineation_set:
                        new_idx = description_valid_idx if combination == [] else np.intersect1d(index,description_valid_idx)
                        new_set.append([combination+[attribute + ':' + description,], new_idx])
                attribute_combineation_set = new_set
            all_attribute_combineation_set += attribute_combineation_set
        # print('len', len(all_attribute_combineation_set))
        non_empty = []
        for a_set in all_attribute_combineation_set:
            a_set[1] = list(set(a_set[1]))
            if len(a_set[1]) > 0:
                non_empty.append(a_set)
        all_attribute_combineation_set = non_empty

        # Compute accuracy, error cover, data cover
        error_cover_list = []
        data_cover = []
        acc_list = []
        all_error = np.sum(self.error_binary_list[data_idx])
        for a_set in all_attribute_combineation_set:
            error_count = np.sum(self.error_binary_list[a_set[1]])
            error_cover_list.append(error_count/all_error)
            data_cover.append(len(a_set[1])/len(data_idx))
            acc_list.append(1 - error_count/len(a_set[1]))
        self.report.record_combinations(all_attribute_combineation_set, error_cover_list, data_cover, acc_list)

        # Greedy algorithm to find combinations to cover ERROR_COVER_THRESHOLD(90% default) errors
        if high_cover_combinations:
            combination, index, error_cover, data_cover, acc_list = [],[],[],[],[]
            cover_rate = 0
            while cover_rate < ERROR_COVER_THRESHOLD:
                error_cover_list = []
                new_indexs = []
                for i, a_set in enumerate(all_attribute_combineation_set):
                    new_index = list(set(index + a_set[1]))
                    error_count = np.sum(self.error_binary_list[new_index])
                    error_cover_list.append(error_count)
                    new_indexs.append(new_index)
                max_cover_id = np.argmax(error_cover_list)
                index = new_indexs[max_cover_id]
                combination.append(all_attribute_combineation_set[max_cover_id][0])
                cover_rate = error_cover_list[max_cover_id]/all_error
                error_cover.append(cover_rate)
                data_cover.append(len(index)/len(data_idx))
                acc_list.append(1 - error_cover_list[max_cover_id]/len(index))

            self.print_output("High Error Coverage Attribute Combinations: ",*combination)
            self.report.record_high_coverage_combinations(combination, error_cover, data_cover, acc_list)

    
    def unlabel_selection(self, error_count=5, cover_threshold=0.9):
        suggestions = self.report.suggestion_for_combinations_by_rate()
        unlabel_idx = np.array([i for i in range(len(self.split)) if self.split[i]==UNLABELED])
        selected = []
        cover = 0
        count = 0
        for suggestion in suggestions:
            count += 1
            if count > error_count and cover > cover_threshold:
                break
            combination = suggestion['combination'][0]
            for i in unlabel_idx:
                match = 0
                for attribute_description in combination:
                    attribute, description = attribute_description.split(':')
                    if self.attributes[attribute]['data'][i] == description:
                        match += 1
                if match == len(combination):
                    selected.append(i)
            selected = list(set(selected))
            cover += suggestion['error_cover']
            if len(selected) > 0:
                print(count,len(suggestions))
                self.print_output('recent data: %.4f'%(len(selected)/len(unlabel_idx)), 'recent error cover: %.4f'%cover, 'acc: %.4f'%suggestion['acc'], 'cover: %.4f'%suggestion['error_cover'])
                self.print_output('actual acc: %.4f'%(1-sum(self.error_binary_list[selected])/len(selected)),'actual error cover: %.4f'%(sum(self.error_binary_list[selected])/ sum(self.error_binary_list[unlabel_idx])))
        return selected
    
    def unlabel_selection_with_confidence(self, confidence, budget, error_count=5, cover_threshold=0.8, distribution_aware=True):
        import tqdm
        suggestions = self.report.suggestion_for_combinations_by_rate()
        suggestions_selected = []
        unlabel_idx = np.array([i for i in range(len(self.split)) if self.split[i]==UNLABELED])
        selected = []
        idxs = []
        cover = 0
        count = 0
        for suggestion in tqdm.tqdm(suggestions):
            count += 1
            if count > error_count and cover > cover_threshold:
                break
            combination = suggestion['combination'][0]
            suggestion_idxs = []
            for i in unlabel_idx:
                match = 0
                for attribute_description in combination:
                    attribute, description = attribute_description.split(':')
                    if self.attributes[attribute]['data'][i] == description:
                        match += 1
                if match == len(combination):
                    selected.append(i)
                    suggestion_idxs.append(i)
            selected = list(set(selected))
            cover += suggestion['error_cover']
            suggestions_selected.append(suggestion)
            idxs.append(suggestion_idxs)
            if len(selected) > 0:
                self.print_output('recent data: %.4f'%(len(selected)/len(unlabel_idx)), 'recent error cover: %.4f'%cover, 'acc: %.4f'%suggestion['acc'], 'cover: %.4f'%suggestion['error_cover'])
                self.print_output('actual acc: %.4f'%(1-sum(self.error_binary_list[selected])/len(selected)),'actual error cover: %.4f'%(sum(self.error_binary_list[selected])/ sum(self.error_binary_list[unlabel_idx])))

        final_selection = []
        sorted_list = np.argsort(confidence)
        if not distribution_aware:
            for i in sorted_list:
                if i in selected:
                    final_selection.append(i)
        else:
            while len(final_selection) < budget: 
                for suggestion,suggestion_idxs in zip(suggestions_selected,idxs):
                    data_length = budget * suggestion['error_cover']/cover
                    count = 0
                    for i in sorted_list:
                        if i in suggestion_idxs and i not in final_selection:
                            final_selection.append(i)
                            count += 1
                            if count >= data_length:
                                break

        return final_selection[:budget]
        
    def date_generation(self, combine_num=5, attribute_num=200, address="exampleData/Imagenet10/generation.json", label_specific=True):
        generation_dict = {}
        for label in range(len(self.label_number)):
            generate_used = [i['combination'][0] for i in MD.report.suggestion_for_combinations_by_rate()[:attribute_num]]
            generate_used = [[i.split(':')[1] for i in g] for g in generate_used]
            generation_dict[label] = generate_used
        f = open(address,'w')
        attributes = json.dumps(generation_dict, indent=4)
        f.write(attributes)
        return attributes