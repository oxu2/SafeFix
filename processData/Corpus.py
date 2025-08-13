from utils import blipFeature, data_utils
import json
import numpy as np

def emptyAttribute():
    return {"data":[],"word":[],"prompt":[],"type":"description"}
    
class AttributeCorpus():
    def __init__(self, datas=[], attribute_dict={}) -> None:
        """
        datas (n*image size) : list of images / list of image addresses 
        attribute_dict ('attribute name': [distinct descriptions of this attribute]): pre-defined attributes
        """
        if type(attribute_dict) == str:
            self.attribute_dict = json.load(open(attribute_dict))
        else:
            self.attribute_dict = attribute_dict
        self.datas = datas
    
    def auto_corpus(self, low_d_cluster=True, feature_num=10000, vqa_num=1000):
        print("Get Features..")
        idxs = np.arange(len(self.datas))
        np.random.shuffle(idxs)
        idxs = idxs[:feature_num]
        datas = [self.datas[i] for i in idxs]
        image_features = blipFeature.get_image_features(datas)
        if low_d_cluster:
            image_features = data_utils.tsne(image_features.detach().cpu().numpy())
            idxs = data_utils.kmeans_selection(image_features, vqa_num)
        else:
            idxs = data_utils.kmeans_selection(image_features, vqa_num)
        vqa_data = [datas[i] for i in idxs]
        for name, attribute in self.attribute_dict.items():
            if attribute["type"] == "description":
                print("Get descriptions for ", name)
                questions = attribute["question"]
                descriptions = blipFeature.question_answering(vqa_data, questions)
                attribute["word"] += descriptions
        for _, item in self.attribute_dict.items():
            item['word'] = list(set(item['word']))
        return self.attribute_dict

    def add_attribute(self, names, types='description'):
        assert type(names)==list
        if type(types) == str:
            types = [types for i in names]
        for name,data_type in zip(names,types):
            self.attribute_dict[name] = emptyAttribute()
            self.attribute_dict[name]['type'] = data_type

    def export_json(self, file="."):
        f = open(file ,'w')
        save_dict = {}
        save_dict['Corpus'] = self.attribute_dict
        f.write(json.dumps(save_dict,indent=4))

