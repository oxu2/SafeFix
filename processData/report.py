import numpy as np
from utils.defines import *
import os
import json

class ReportGenerator():
    def __init__(self, attribute_info_dict, train_dis, valid_dis, train_cm, valid_cm, valid_acc, label_names_dict) -> None:
        self.attribute_info_dict = attribute_info_dict
        self.label_names_dict = label_names_dict
        self.labels = [int(label) for label in list(label_names_dict.keys())]
        self.error_list = []
        self.error_dict = {'error':{},}
        self.error_dict['info'] = {}
        self.error_dict['info']['confusion matrix'] = {}
        
        self.error_dict['info']['confusion matrix'][TRAIN] = {label:{predict_label:train_cm[label][predict_label] for predict_label in self.labels} for label in self.labels}
        self.error_dict['info']['confusion matrix'][VALID] = {label:{predict_label:valid_cm[label][predict_label] for predict_label in self.labels} for label in self.labels}
        self.error_dict['info']['distribution'] = {}
        self.error_dict['info']['distribution'][TRAIN] = {self.labels[i]:train_dis[i] for i in range(len(self.labels))}
        self.error_dict['info']['distribution'][VALID] = {self.labels[i]:valid_dis[i] for i in range(len(self.labels))}

        self.error_dict['info']['valid acc'] = valid_acc
        self.error_dict['info']['valid label acc'] = {}
        for label in self.labels:
            self.error_dict['info']['valid label acc'][label] = self.error_dict['info']['confusion matrix'][VALID][label][label]/self.error_dict['info']['distribution'][VALID][label]
        self.error_dict['error']['rare label'] = []
        self.error_dict['error']['rare description'] = []
        self.error_dict['error']['prediction correlation'] = []
        self.error_dict['error']['failure correlation'] = []
        self.error_dict['error']['distribution shift'] = []
        self.error_dict['error']['hard case'] = []

    def record_rare_label_error(self, label):
        error_count = self.error_dict['info']['distribution'][VALID][label] * (1-self.error_dict['info']['valid label acc'][label])
        error_count = int(error_count)
        distribution = self.error_dict['info']['distribution'][VALID][label]
        self.error_dict['error']['rare label'].append({'label':label,'valid distribution':distribution,'valid acc':self.error_dict['info']['valid label acc'][label]})
        self.error_list.append([error_count, self.error_dict['error']['rare label'][-1]])

    def record_rare_case_error(self, attribute, description, distribution_shift=False, is_label=False, is_rare=False):
        error_count = self.attribute_info_dict[attribute]['distribution'][VALID][description] * (1-self.attribute_info_dict[attribute]['valid acc'][description])
        error_count = int(error_count)
        distribution = {description:self.attribute_info_dict[attribute]['distribution'][VALID][description] for description in self.attribute_info_dict[attribute]['word']}
        if is_rare:
            self.error_dict['error']['rare description'].append({'attribute':attribute,'description':description,'valid distribution':distribution,'valid acc':self.attribute_info_dict[attribute]['valid acc'][description], 'train distribution':self.attribute_info_dict[attribute]['distribution'][TRAIN][description]})
            self.error_list.append([error_count, 'rare description', self.error_dict['error']['rare description'][-1]])
        elif distribution_shift:
            self.error_dict['error']['distribution shift'].append({'attribute':attribute,'description':description,'valid distribution':distribution,'valid acc':self.attribute_info_dict[attribute]['valid acc'][description], 'train distribution':self.attribute_info_dict[attribute]['distribution'][TRAIN][description]})
            self.error_list.append([error_count, 'distribution shift', self.error_dict['error']['distribution shift'][-1]])
        elif is_label:
            self.error_dict['error']['rare sub label'].append({'attribute':attribute,'description':description,'valid distribution':distribution,'valid acc':self.attribute_info_dict[attribute]['valid acc'][description], 'train distribution':self.attribute_info_dict[attribute]['distribution'][TRAIN][description]})
            self.error_list.append([error_count, 'rare sub label', self.error_dict['error']['rare sub label'][-1]])
        else:
            self.error_dict['error']['hard case'].append({'attribute':attribute,'description':description,'valid distribution':distribution,'valid acc':self.attribute_info_dict[attribute]['valid acc'][description], 'train distribution':self.attribute_info_dict[attribute]['distribution'][TRAIN][description]})
            self.error_list.append([error_count, 'hard case', self.error_dict['error']['hard case'][-1]])

    def record_correlation_error(self, attribute, description, label_distribution, in_failure=False):
        error_count = self.attribute_info_dict[attribute]['distribution'][VALID][description] * (1-self.attribute_info_dict[attribute]['valid acc'][description]) * np.max(label_distribution)
        error_count = int(error_count)
        correlated_label = np.argmax(label_distribution)
        distribution = {self.labels[i]:label_distribution[i] for i in range(len(self.labels))}
        if not in_failure:
            self.error_dict['error']['prediction correlation'].append({'attribute':attribute,'description':description,'correlated label': int(correlated_label), 'prediction distribution':distribution,'valid acc':self.attribute_info_dict[attribute]['valid acc']})
            self.error_list.append([error_count, 'prediction correlation', self.error_dict['error']['prediction correlation'][-1]])
        else:
            self.error_dict['error']['failure correlation'].append({'attribute':attribute,'description':description,'correlated label': int(correlated_label),'prediction distribution':distribution,'valid acc':self.attribute_info_dict[attribute]['valid acc']})
            self.error_list.append([error_count, 'failure prediction correlation', self.error_dict['error']['failure correlation'][-1]])

    def record_high_coverage_combinations(self, combinations, error_cover, data_cover, acc_list):
        self.error_dict['high error coverage'] = {'combination':combinations, 'error_cover':error_cover, 'data_cover':data_cover, 'acc':acc_list}

    def record_combinations(self, combinations,error_cover, data_cover, acc_list):
        self.error_dict['combinations'] = []
        for combination, error_cov, data_cov, acc in zip(combinations,error_cover, data_cover, acc_list):
            combination[1] = list(combination[1])
            self.error_dict['combinations'].append({'combination':combination, 'error_cover':float(error_cov), 'data_cover':float(data_cov), 'acc':float(acc)})
            

    def suggestion_for_error_list(self, suggestions_num=3):
        self.error_list.sort(key=lambda x:x[0],reverse=True)
        error_list  = self.error_list
        suggestions = []
        for error in error_list:
            if len(suggestions) >= suggestions_num:
                break
            if 'correlation' in error[1]:
                attribute = ['LABEL', error[2]['attribute']]
                description = [[i for i in self.labels if int(i)!=int(error[2]['correlated label'])],error[2]['description'] ]
            else:
                attribute, description = [error[2]['attibute'],], [error[2]['description'],]
            suggestions.append([attribute, description])
        return suggestions

    def suggestion_for_combinations_by_score(self):
        combinations  = self.error_dict['combinations']
        scores = []
        for combination in combinations:
            score = combination['error_cover'] * (combination['acc'] - self.error_dict['info']['valid acc'])
            scores.append(score)
        return [combinations[i] for i in np.argsort(scores)]

    def suggestion_for_combinations_by_rate(self):
        combinations  = self.error_dict['combinations']
        scores = []
        for combination in combinations:
            score = combination['acc']
            scores.append(score)
        return [combinations[i] for i in np.argsort(scores)]

    def save_report(self, dir):
        self.error_list.sort(key=lambda x:x[0],reverse=True)
        # self.error_dict['combinations'].sort(key=lambda x:x['0'],reverse=True)
        self.error_dict['list'] = self.error_list
        if not os.path.exists(dir):
            os.makedirs(dir)
        f = open(dir + '/error_dict.json', 'w')
        errors = json.dumps(self.error_dict,indent=4)
        f.write(errors)
        f = open(dir + '/attribute_info.json', 'w')
        attributes = json.dumps(self.attribute_info_dict,indent=4)
        f.write(attributes)
