import json
from utils.blipFeature import get_img_text_similarity, question_answering, question_answering_list
import numpy as np
import copy
UNCERTAIN_THRESHOLD = 0.2

class AttributeSelection():
    def __init__(self, datas, labels, corpus, label_names_dict={}) -> None:
        self.datas = datas
        self.labels = labels
        self.label_names_dict = label_names_dict if label_names_dict!={} else {label:str(label) for label in set(labels)}
        self.data_length = len(self.datas)
        if type(corpus) == str:
            self.corpus = json.load(open(corpus))["Corpus"]
            self.attributes_dict = copy.deepcopy(self.corpus)
        else:
            self.attributes_dict = corpus

    def match_description_to_data(self):
        for attribute in self.attributes_dict.keys():
            print("Match ",attribute)
            self.attributes_dict[attribute]['data'] = ["" for i in range(self.data_length)]
            self.attributes_dict[attribute]['uncertain_idx'] = []
            if type(self.attributes_dict[attribute]['prompt']) == str:
                self.attributes_dict[attribute]['prompt'] = [self.attributes_dict[attribute]['prompt'],]
            # if self.attributes_dict[attribute]["type"] == 'binary':
            #     print("Binary")
            #     self.match_binary_description(attribute)
            elif self.attributes_dict[attribute]["type"] == 'vqa' or self.attributes_dict[attribute]["type"] == 'binary':
                print("VQA")
                self.match_vqa_description(attribute)
            else:
                print("Description match")
                self.match_general_description(attribute)

    def match_vqa_description(self, attribute):
        question = self.attributes_dict[attribute]["question"]
        if '#LABEL' in question:
            questions = []
            for i in range(self.data_length):
                label_name = self.label_names_dict[self.labels[i]]
                label_question = question.replace("#LABEL",label_name)
                questions.append(label_question)
            answer = question_answering_list(self.datas, questions)
        else:
            answer = question_answering(self.datas, question)
        self.attributes_dict[attribute]["data"] = answer

    def match_binary_description(self, attribute, by_vqa=False):
        indexs = [i for i in range(self.data_length)]
        prompt1, prompt2 = self.attributes_dict[attribute]["prompt"]
        similarity = get_img_text_similarity([self.datas[i] for i in indexs], [prompt1, prompt2])
        similarity = similarity.detach().cpu().numpy()
        selected_attribute_index = np.argmax(similarity,axis=1)
        idxs = self.uncertain_data_idx(similarity)[0]
        uncertain_idx = [indexs[i] for i in idxs]
        for i, idx in enumerate(indexs):
            self.attributes_dict[attribute]['data'][idx] = self.attributes_dict[attribute]["word"][selected_attribute_index[i]]
        self.attributes_dict[attribute]['uncertain_idx'] = uncertain_idx

    def similarity_match(self, attribute, indexs, label=None):
        similaritys = []
        uncertain_idx = []
        for prompt in self.attributes_dict[attribute]["prompt"]:
            sentences = self.generate_prompts(attribute, prompt, label)
            similarity = get_img_text_similarity([self.datas[i] for i in indexs], sentences)
            similarity = similarity.detach().cpu().numpy()
            similaritys.append(similarity)
        similaritys = np.sum(similaritys,axis=0)
        print(similaritys.shape)
        selected_attribute_index = np.argmax(similaritys,axis=1)
        if similaritys.shape[1] > 1:
            idxs = self.uncertain_data_idx(similaritys)[0]
            uncertain_idx += [indexs[i] for i in idxs]
        for i, idx in enumerate(indexs):
            self.attributes_dict[attribute]['data'][idx] = self.attributes_dict[attribute]["word"][selected_attribute_index[i]]
        self.attributes_dict[attribute]['uncertain_idx'] += uncertain_idx
    
    def match_general_description(self, attribute):
        need_partition = False
        for prompt in self.attributes_dict[attribute]['prompt']:
            if "#LABEL" in prompt:
                need_partition = True
        if need_partition:
            for label in self.label_names_dict.keys():
                indexs = [i for i in range(self.data_length) if self.labels[i]==label]
                self.similarity_match(attribute, indexs, label)
        else:
            indexs = [i for i in range(self.data_length)]
            self.similarity_match(attribute, indexs)

    def generate_prompts(self, attribute, prompt, label=None):
        label_name = self.label_names_dict[label] if label is not None else ""
        descriptions = self.attributes_dict[attribute]["word"]
        sentences = []
        for description in descriptions:
            sentence = prompt.replace("#1",description).replace("#LABEL",label_name)
            sentences.append(sentence)
        return sentences

    def save_attributes(self, file):
        f = open(file, 'w')
        attributes = json.dumps(self.attributes_dict,indent=4)
        f.write(attributes)

    def uncertain_data_idx(self, similarity):
        similarity_sort = np.sort(similarity,axis=1)[:,::-1]
        U = similarity_sort[:, 0] - similarity_sort[:,1]
        return np.where(U<UNCERTAIN_THRESHOLD)

