import numpy as np
from utils.defines import *
import matplotlib.pyplot as plt
from PIL import Image

def get_correctness(predictions, labels):
    correctness = []
    for i, j in zip(predictions, labels):
        if j != '' and i!= '':
            if int(i) == int(j):
                correctness.append(CORRECT)
            else:
                correctness.append(WRONG)
        else:
            correctness.append(UNKNOWN)
    return correctness

def get_scatter_data():
    pass

def show_image(img):
    if type(img) == str:
        import os
        if os.path.exists(img):
            img = Image.open(img)
            plt.axis("off")
            plt.imshow(img)
            plt.show()
        else:
            print(img)
    else:
        if img.shape[0] == 1:
            img = img.reshape([img.shape[0],img.shape[1]])
            plt.imshow(img, cmap='gray')
        elif len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.show()
    return img

class data_filer():
    def __init__(self,predictions,labels,tags,attributes_dict) -> None:
        self.predictions = predictions
        self.data_length = len(tags)
        self.update_attributes_dict(attributes_dict)
        self.labels = labels
        self.correctness = get_correctness(predictions,labels)
        self.tags = tags    
        self.index = np.arange(len(predictions))
    
    def update_attributes_dict(self,attributes_dict):
        self.attributes_dict = attributes_dict
        attributes_list = [attributes_dict[attribute]['data'] for attribute in attributes_dict.keys()]
        self.attributes_list = [[attributes_list[i][j] for i in range(len(attributes_list))] for j in range(self.data_length)]

    def attr_match(self,selected_attrs, node_attrs):
        match = 0
        for attr, node_attr in zip(selected_attrs, node_attrs):
            if node_attr in attr or ALL in attr:
                match += 1
        if match == len(node_attrs):
            return True
        else: 
            return False

    def dataset_match(self,selected_dataset, tag):
        if selected_dataset == ALL:
            return True
        else:
            return selected_dataset==tag
    
    def data_match(self,selected_data, correctness, i, selected_data_index, selected_attr=None, recent_attribute=None):
        if selected_data == ALL:
            return True
        elif selected_data == 'Failure':
            return correctness == WRONG
        elif selected_data == 'Uncertain Data':
            if recent_attribute in  self.attributes_dict.keys():
                return i in self.attributes_dict[recent_attribute]['uncertain_idx']
            else:
                return True
        elif selected_data == 'Selected':
            return i in selected_data_index
        elif selected_data == 'Unselected':
            return i not in selected_data_index

    def filter(self, selected_dataset=None, selected_data=None, selected_attr=None, selected_data_index=None, indexs=None, recent_attribute=None):
        if indexs is None: indexs = self.index
        selected_indexs = []
        for i in indexs:
            if selected_dataset is None or self.dataset_match(selected_dataset,self.tags[i]):
                if selected_data is None or self.data_match(selected_data,self.correctness[i], i, selected_data_index, selected_attr): 
                    if selected_attr is None or self.attr_match(selected_attr,self.attributes_list[i], recent_attribute=recent_attribute): 
                        selected_indexs.append(i)
        return selected_indexs

    def get_attribute(self, target, indexs=None):
        if indexs is None: indexs = self.index
        if target=='Prediction':
            labels = [self.predictions[i] for i in indexs]
        elif target=='Correctness':
            labels = [self.correctness[i] for i in indexs]
        elif target=='Dataset':
            labels = [self.tags[i] for i in indexs]
        else:
            for attr_index, name in enumerate(self.attributes_dict.keys()):
                if name == target:
                    labels = [self.attributes_list[i][attr_index] for i in indexs]
                    break
        return labels

def get_colors(labels):
    label = list(set(labels))
    import matplotlib.colors as mcolors
    color_list = list(mcolors.CSS4_COLORS.values())[2:]
    colors = []
    for i in labels:
        for ci, j in enumerate(label):
            if i == j:
                colors.append(color_list[ci])
                break
    return colors

def get_data_attr_distribution(attrs):
    length = len(attrs)
    distinct_attrs = list(set(attrs))
    count_attrs = [0 for i in distinct_attrs]
    attr_to_ids = dict([(attr,j) for j,attr in enumerate(distinct_attrs)])
    for i, attr in enumerate(attrs):
        if length!=0: count_attrs[attr_to_ids[attr]] += 1/length
        else: count_attrs[attr_to_ids[attr]] += 1
    distinct_attrs = [str(i) for i in distinct_attrs]
    return distinct_attrs, count_attrs
