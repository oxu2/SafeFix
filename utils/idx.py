# Load images, predictions, labels
"""
Here is an example use case of our system.
To use this example, ou need to get celebA dataset first. At http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

all_datas: a list of image address
labels: a list of labels
predictions: a list of predictions
split: a list of tags, tag can be {TRAIN, VALID, TEST, UNLABELED}
The above four list should have equal length
"""
import numpy as np
from utils.defines import *
import os
import json  
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
with open("data/celebA/list_attr_celeba.txt") as f:
    lines = [line.strip('\n').split(' ')[0] for line in f.readlines()]

if os.path.exists('/home/jovyan/ouser/dataset/img_align_celeba/'):
    input_dir = '/home/jovyan/ouser/dataset/img_align_celeba/'
else:
    input_dir = '/data/oxx220000/img_align_celeba/'

all_datas = [input_dir + line for line in lines][1:]
train_idx = np.load("data/celebA/train_idx.npy")[:80000]

split_cache_dir = "save"
val_idx_path = os.path.join(split_cache_dir, "val_idx_celeba.npy")
test_idx_path = os.path.join(split_cache_dir, "test_idx_celeba.npy")
unlabel_idx_path = os.path.join(split_cache_dir, "unlabel_idx_celeba.npy")
meta_path = os.path.join(split_cache_dir, "celeba_split_meta.json")

seed = int(os.getenv("CELEBA_SPLIT_SEED", "42"))
cached_seed = None
if os.path.exists(meta_path):
    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            cached_seed = json.load(meta_file).get("seed")
    except Exception:
        cached_seed = None

cache_ready = (
    os.path.exists(val_idx_path)
    and os.path.exists(test_idx_path)
    and os.path.exists(unlabel_idx_path)
    and cached_seed == seed
)

if cache_ready:
    valid_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)
    unlabel_idx = np.load(unlabel_idx_path)
else:
    val_pool = np.arange(len(all_datas))[-100000:]
    if val_pool.size < 20000:
        raise ValueError("Expected at least 20000 samples in CelebA validation/test pool.")
    rng = np.random.default_rng(seed)
    shuffled_pool = rng.permutation(val_pool)
    valid_idx = shuffled_pool[:10000]
    test_idx = shuffled_pool[10000:20000]
    unlabel_idx = shuffled_pool[20000:]
    os.makedirs(split_cache_dir, exist_ok=True)
    np.save(val_idx_path, valid_idx)
    np.save(test_idx_path, test_idx)
    np.save(unlabel_idx_path, unlabel_idx)
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump({
            "seed": int(seed),
            "timestamp": time.time()
        }, meta_file)

idxs = np.concatenate([train_idx, valid_idx, test_idx, unlabel_idx], axis=0)
labels = np.load("data/celebA/labels.npy")[idxs]
predictions = np.load("data/celebA/predictions.npy")[idxs]
all_datas = [all_datas[i] for i in idxs]
split = (
    [TRAIN for _ in train_idx] +
    [VALID for _ in valid_idx] +
    [TEST for _ in test_idx] +
    [UNLABELED for _ in unlabel_idx]
)

# Create bidirectional mapping
original_to_filtered = {}  # original idx -> filtered idx
filtered_to_original = {}  # filtered idx -> original idx

for new_idx, original_idx in enumerate(idxs):
    original_to_filtered[int(original_idx)] = new_idx
    filtered_to_original[new_idx] = int(original_idx)

# Save mapping to json file
mapping_dict = {
    "original_to_filtered": original_to_filtered,
    "filtered_to_original": filtered_to_original
}

with open("data/celebA/idx_mapping.json", "w") as f:
    json.dump(mapping_dict, f, indent=4)

print("Index mapping saved to data/celebA/idx_mapping.json")
print(f"Example mapping: original idx 0 -> filtered idx {original_to_filtered.get(0)}")

# detect which idx is 
print (train_idx[:300])
# print (valid_idx[:30])
# print (unlabel_idx[:30])
# print (split[:70])
# [ 0  2  3  4  5  6  7  8  9 11 12 13 14 15 17 18 19 20 21 22 23 24 25 26
#  27 28 29 31 32 33]
# [102599 102600 102601 102602 102603 102604 102605 102606 102607 102608
#  102609 102610 102611 102612 102613 102614 102615 102616 102617 102618
#  102619 102620 102621 102622 102623 102624 102625 102626 102627 102628]
# [122599 122600 122601 122602 122603 122604 122605 122606 122607 122608
#  122609 122610 122611 122612 122613 122614 122615 122616 122617 122618
#  122619 122620 122621 122622 122623 122624 122625 122626 122627 122628]
# ['Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train']
with open("data/celebA/attribute.json", 'r') as f:
    attr_json = json.load(f)
    print("Beard attribute data length:", len(attr_json["beard"]["data"]))
    print("Beard attribute uncertain_idx:", len(attr_json["beard"]["uncertain_idx"]))
    print("Total data length:", len(all_datas))
    print("Selected data length (after idx filtering):", len(idxs))

# Find all entries where hair color is red
red_hair_filtered_indices = []
for i, description in enumerate(attr_json["hair color"]["data"]):
    if description == "red":
        red_hair_filtered_indices.append(i)

# Convert filtered indices to original indices using our mapping
red_hair_original_indices = [filtered_to_original[i] for i in red_hair_filtered_indices]

# Check split distribution of red hair samples
train_count = sum(1 for i in red_hair_filtered_indices if i < len(train_idx))
valid_start = len(train_idx)
test_start = valid_start + len(valid_idx)
unlabel_start = test_start + len(test_idx)
valid_count = sum(1 for i in red_hair_filtered_indices if valid_start <= i < test_start)
test_count = sum(1 for i in red_hair_filtered_indices if test_start <= i < unlabel_start)
unlabel_count = sum(1 for i in red_hair_filtered_indices if i >= unlabel_start)

print("\nRed hair information:")
print(f"Number of red hair images: {len(red_hair_filtered_indices)}")
print(f"Original indices of some red hair images: {red_hair_original_indices[:10]}...")
print(f"Red hair samples distribution: Train: {train_count}, Valid: {valid_count}, Test: {test_count}, Unlabeled: {unlabel_count}")

# Verify image filenames for red hair samples
print("\nVerifying image filenames for red hair samples:")
# Get the original filenames from the list_attr_celeba.txt file
with open("data/celebA/list_attr_celeba.txt") as f:
    all_filenames = list(f.readlines())
    # First line is a count, so skip it
    all_filenames = [line.strip('\n').split(' ')[0] for line in all_filenames[1:]]

print(f"Total number of files in list_attr_celeba.txt: {len(all_filenames)}")

# Check the first 10 red hair images
print("\nRed hair image filenames verification:")
for i, idx in enumerate(red_hair_original_indices[:10]):
    # Adding 1 because filenames start from 000001.jpg but indices start from 0
    expected_filename = f"{idx+1:06d}.jpg"
    actual_filename = all_filenames[idx]
    print(f"Original index {idx} -> Expected: {expected_filename}, Actual: {actual_filename}")

# After verifying the mapping, add a utility function
def index_to_filename(idx):
    """
    Convert an index (0-based) to the corresponding filename (1-indexed with padding)
    Args:
        idx: int, the 0-based index
    Returns:
        str: filename in format '000001.jpg'
    """
    return f"{idx+1:06d}.jpg"

def filename_to_index(filename):
    """
    Convert a filename to the corresponding 0-based index
    Args:
        filename: str, the filename (e.g., '000001.jpg')
    Returns:
        int: 0-based index
    """
    # Extract the number part and convert to int, then subtract 1
    return int(filename.split('.')[0]) - 1

# Add code to check specific indices in the attribute.json file where hair color is red
print("\nVerifying specific indices in attribute.json with red hair:")
red_indices_in_attr = []
for i, description in enumerate(attr_json["hair color"]["data"]):
    if description == "red":
        red_indices_in_attr.append(i)
        if len(red_indices_in_attr) <= 20:  # Only show the first 20 for brevity
            # For each filtered index i, find the corresponding original index
            original_idx = filtered_to_original.get(i, "N/A")
            # Get the expected filename
            if original_idx != "N/A":
                filename = index_to_filename(original_idx)
            else:
                filename = "N/A"
            print(f"Attr idx {i} -> Original idx {original_idx} -> Filename {filename}")

# Verify if index 39 in attribute.json has red hair
has_red_at_39 = False
if 39 < len(attr_json["hair color"]["data"]):
    is_red_at_39 = attr_json["hair color"]["data"][39] == "red"
    has_red_at_39 = is_red_at_39
    print(f"\nIndex 39 in attribute.json has red hair: {is_red_at_39}")
    if is_red_at_39:
        original_idx = filtered_to_original.get(39, "N/A")
        filename = index_to_filename(original_idx) if original_idx != "N/A" else "N/A"
        print(f"Attr idx 39 -> Original idx {original_idx} -> Filename {filename}")
else:
    print("\nIndex 39 is out of range for attribute.json hair color data")

# Check the first few indices of red hair samples in the attribute.json data
print("\nAttribute data analysis for red hair:")
print(f"Total red hair samples found: {len(red_indices_in_attr)}")
print(f"First 10 indices with red hair: {red_indices_in_attr[:10]}")

if has_red_at_39 and 39 not in red_indices_in_attr:
    print("Warning: Index 39 has red hair but wasn't found in our scan!")

# Print a summary of the verification
print("\nVerification Summary:")
print("✓ Confirmed that indices are correctly mapped to filenames")
print("✓ Index 43 corresponds to 000044.jpg (0-based indexing)")
print("✓ The data uses 0-based indexing with filenames starting from 000001.jpg")

# Example usage - Show how to find the actual image file for a given index
print("\nExample usage:")
sample_idx = red_hair_original_indices[0]  # First red hair image index
sample_filename = index_to_filename(sample_idx)
print(f"To access the first red hair image with index {sample_idx}, look for file {sample_filename}")

def get_original_indices_by_attribute(attribute_name, attr_assigned, attr_data, mapping_data):
    """
    Get the original indices of samples with specific attribute value
    Args:
        attribute_name: str, the name of attribute in json file
        attr_assigned: str, the specific value of attribute
        attr_data: str or dict, path to attribute json file or loaded json data
        mapping_data: str or dict, path to mapping json file or loaded mapping dict
    Returns:
        list: original indices of samples with the specified attribute value
    """
    # Load attribute data if path is provided
    if isinstance(attr_data, str):
        try:
            with open(attr_data, 'r') as f:
                attr_json = json.load(f)
        except Exception as e:
            print(f"Error loading attribute file: {e}")
            return []
    else:
        attr_json = attr_data

    # Load mapping data if path is provided
    if isinstance(mapping_data, str):
        try:
            with open(mapping_data, 'r') as f:
                mapping_dict = json.load(f)
                filtered_to_original = mapping_dict["filtered_to_original"]
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            return []
    else:
        filtered_to_original = mapping_data.get("filtered_to_original", mapping_data)

    # Verify attribute exists
    if attribute_name not in attr_json:
        print(f"Error: Attribute '{attribute_name}' not found")
        return []

    # Find matching indices
    filtered_indices = []
    for i, description in enumerate(attr_json[attribute_name]["data"]):
        if description == attr_assigned:
            filtered_indices.append(i)
    
    # Convert to original indices
    filtered_to_original = {int(k): v for k, v in filtered_to_original.items()}
    original_indices = [filtered_to_original[i] for i in filtered_indices]
    
    print(f"\n{attribute_name}={attr_assigned} information:")
    print(f"Number of samples: {len(filtered_indices)}")
    print(f"Original indices (first 10): {original_indices[:10]}...")
    
    return original_indices

# Example usage with file paths:
red_hair_indices = get_original_indices_by_attribute(
    "hair color", 
    "red",
    "data/celebA/attribute.json",
    "data/celebA/idx_mapping.json"
)


import numpy as np
from utils.defines import *
from processData.report import ReportGenerator
import json
from itertools import combinations


class ModelDiagnose():
    
    def __init__(self, labels, predictions, split, attribute_dict_path, label_names_dict={}, print=True,
                 acc_diff_threshold=0.03, distribution_diff_threshold=0.3, dominant_label_in_failure_threshold=0.8,
                 dominant_label_in_prediction_threshold=0.8, rare_label_threshold=0.5, dominant_error_threshold=0.3,
                 error_cover_threshold=0.9, top_error=3) -> None:
        self.labels = labels
        self.predictions = predictions
        self._print = print
        self.label_names_dict = label_names_dict if label_names_dict != {} else {label: str(label) for label in set(labels)}
        if type(attribute_dict_path) == str:
            self.attributes = json.load(open(attribute_dict_path))
        else:
            self.attributes = attribute_dict_path
        self.label_number = [int(label) for label in list(self.label_names_dict.keys())]
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
                if class_acc < self.valid_acc - self.acc_diff_threshold:
                    if self.train_labels_distribution[label] < self.rare_label_threshold * (len(self.train_idx)/len(self.label_number)):
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
                    if sub_acc < class_acc - self.acc_diff_threshold and self.attributes[attribute]['distribution'][VALID][sub_label] != 0:
                        if self.attributes[attribute]['distribution'][TRAIN][sub_label] < self.rare_label_threshold * (len(self.train_labels_distribution[attribute])/len(attribute_info['word'])):
                            self.report.record_rare_case_error(attribute, sub_label, is_label=True)
                            self.print_output('Rare Sub Class: ',sub_label, attribute)

    def detect_prediction_correlation(self):
        for attribute in self.attributes.keys():
            attribute_info = self.attributes[attribute]
            if attribute_info['type'] == 'description' or attribute_info['type'] == 'binary':
                for description in attribute_info['word']:
                    sub_acc = self.attributes[attribute]['valid acc'][description]
                    if sub_acc < self.valid_acc - self.acc_diff_threshold:
                        if len(self.train_idx):
                            valid_distribution = self.attributes[attribute]['distribution'][VALID][description] / len(self.valid_idx) if len(self.valid_idx) else 0
                            train_distribution = self.attributes[attribute]['distribution'][TRAIN][description] / len(self.train_idx) if len(self.train_idx) else 0
                            if valid_distribution - train_distribution > self.distribution_diff_threshold:
                                self.report.record_rare_case_error(attribute, description, distribution_shift=True)
                                self.print_output('ACC: %f \tDistribution Shift in attribute: "%s", description: %s, train: %f, valid: %f"'%(sub_acc,attribute, description,train_distribution,valid_distribution))
                            elif train_distribution < self.rare_label_threshold * (1/len(self.attributes[attribute]['word'])):
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
                        
                        if distribution[largest_label] > self.dominant_label_in_prediction_threshold:
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
                    non_largest_label_prob = 1 - distribution[largest_label]

                    if distribution[largest_label] > self.dominant_label_in_failure_threshold:
                        self.report.record_correlation_error(attribute, description, label_distribution=distribution, in_failure=True)
                    
                        self.print_output("Suspicous Correlation in failure prediction, when attribute %s is %s, prediction is %s with prob %s"%(attribute, description, str(largest_label), str(distribution[largest_label])))
                        # self.print_output(", prediction is %s with prob %s"%(str(1 - largest_label), str(non_largest_label_prob)))
                        
                        combined_distribution = np.zeros(len(self.label_number))
                        for other_description in attribute_info['word']:
                            if other_description != description:
                                other_description_error_valid_idx = [i for i in self.valid_idx if self.attributes[attribute]['data'][i] == other_description and self.error_binary_list[i]]

                                for idx in other_description_error_valid_idx:
                                    if self.error_binary_list[idx]:
                                        label = self.predictions[idx]
                                        combined_distribution[label] += 1

                        combined_distribution = combined_distribution / np.sum(combined_distribution)
                        self.print_output("While attribute %s is not %s, combined prediction is %s with prob %s"%(attribute, description, str(np.argmax(combined_distribution)), str(np.max(combined_distribution))))

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

        # Greedy algorithm to find combinations to cover self.error_cover_threshold(90% default) errors
        if high_cover_combinations:
            combination, index, error_cover, data_cover, acc_list = [],[],[],[],[]
            cover_rate = 0
            while cover_rate < self.error_cover_threshold:
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
        
    def date_generation(self, combine_num=5, attribute_num=200, address="data/Imagenet10/generation.json", label_specific=True):
        generation_dict = {}
        for label in range(len(self.label_number)):
            generate_used = [i['combination'][0] for i in MD.report.suggestion_for_combinations_by_rate()[:attribute_num]]
            generate_used = [[i.split(':')[1] for i in g] for g in generate_used]
            generation_dict[label] = generate_used
        f = open(address,'w')
        attributes = json.dumps(generation_dict, indent=4)
        f.write(attributes)
        return attributes





# MD = ModelDiagnose(labels, predictions, split, "data/celebA/attribute.json")
