import numpy as np
from utils.defines import *
import os

# Ensure script runs under the 'hibug' conda environment
# if os.environ.get('CONDA_DEFAULT_ENV') != 'hibug':
#     print("Activating the 'hibug' conda environment...")
#     os.system("conda activate hibug")

# Output filename for labels
root_dir = 'imagenet_subset'
output_file = 'labels10.npy'

# Prepare lists for labels and image paths
all_datas = []
labels = []

# Walk through each subfolder
for class_name in sorted(os.listdir(root_dir)):
    class_dir = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    # For every file in the class folder
    for fname in os.listdir(class_dir):
        # Optionally filter by image extensions
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            labels.append(class_name)
            all_datas.append(os.path.join(class_dir, fname))

# Convert to a NumPy array and save
labels_array = np.array(labels)
np.save(output_file, labels_array)
print(f'Saved {labels_array.shape[0]} labels to {output_file}')

# === Attribute selection ===
import json
import copy
from utils.blipFeature import get_img_text_similarity, question_answering, question_answering_list

UNCERTAIN_THRESHOLD = 0.2

class AttributeSelection:
    def __init__(self, datas, labels, corpus, label_names_dict=None):
        self.datas = datas
        self.labels = labels
        self.label_names_dict = (
            label_names_dict
            if label_names_dict is not None
            else {label: str(label) for label in set(labels)})
        self.data_length = len(self.datas)
        if isinstance(corpus, str):
            self.attributes_dict = json.load(open(corpus))["Corpus"]
        else:
            self.attributes_dict = copy.deepcopy(corpus)

    def match_description_to_data(self):
        for attribute in self.attributes_dict.keys():
            print("Match", attribute)
            self.attributes_dict[attribute]['data'] = [""] * self.data_length
            self.attributes_dict[attribute]['uncertain_idx'] = []
            prompts = self.attributes_dict[attribute]['prompt']
            if isinstance(prompts, str):
                prompts = [prompts]
            if self.attributes_dict[attribute]['type'] in ('vqa', 'binary'):
                self.match_vqa_description(attribute)
            else:
                self.match_general_description(attribute)

    def match_vqa_description(self, attribute):
        question = self.attributes_dict[attribute]['question']
        if '#LABEL' in question:
            questions = []
            for i in range(self.data_length):
                label_name = self.label_names_dict[self.labels[i]]
                questions.append(question.replace("#LABEL", label_name))
            answer = question_answering_list(self.datas, questions)
        else:
            answer = question_answering(self.datas, question)
        self.attributes_dict[attribute]['data'] = answer

    def similarity_match(self, attribute, indexs, label=None):
        sims = []
        for prompt in self.attributes_dict[attribute]["prompt"]:
            sentences = [
                prompt.replace("#1", desc).replace(
                    "#LABEL", self.label_names_dict.get(label, ""))
                for desc in self.attributes_dict[attribute]["word"]
            ]
            sim = get_img_text_similarity(
                [self.datas[i] for i in indexs], sentences
            ).detach().cpu().numpy()
            sims.append(sim)
        sims = np.sum(sims, axis=0)
        selected = np.argmax(sims, axis=1)
        if sims.shape[1] > 1:
            diffs = np.sort(sims, axis=1)[:, ::-1]
            U = diffs[:, 0] - diffs[:, 1]
            uncertain = np.where(U < UNCERTAIN_THRESHOLD)[0]
            self.attributes_dict[attribute]['uncertain_idx'].extend(
                [indexs[i] for i in uncertain])
        for idx_i, img_idx in enumerate(indexs):
            self.attributes_dict[attribute]['data'][img_idx] = \
                self.attributes_dict[attribute]["word"][selected[idx_i]]

    def match_general_description(self, attribute):
        if any("#LABEL" in p for p in self.attributes_dict[attribute]['prompt']):
            for label in set(self.labels):
                idxs = [i for i, lbl in enumerate(self.labels) if lbl == label]
                self.similarity_match(attribute, idxs, label)
        else:
            self.similarity_match(attribute, list(range(self.data_length)))

    def save_attributes(self, file):
        with open(file, 'w') as f:
            json.dump(self.attributes_dict, f, indent=4)

# Run attribute selection
corpus_path = 'imagenet10_corpus.json'
AS = AttributeSelection(all_datas, labels, corpus_path)
AS.match_description_to_data()
AS.save_attributes('attribute10.json')
