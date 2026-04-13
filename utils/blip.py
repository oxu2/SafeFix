import argparse
import copy
import csv
import json
import os
import random
import shlex
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.defines import *


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
DEFAULT_FOLDER_ROOT = 'data/imagenet_subset'
DEFAULT_COCO_ROOT = '/data/oxx220000/coco'
DEFAULT_COCO_POSE_CORPUS = 'save/corpus_coco_pose_gemini.json'
DEFAULT_BLIP_ENV = 'ad'


def get_blip_env_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--blip_env' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith('--blip_env='):
            return arg.split('=', 1)[1]
    return DEFAULT_BLIP_ENV


def ensure_conda_env(env_name):
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env == env_name:
        return

    print(f"Switching to conda environment '{env_name}' (current: {current_env or 'None'})")
    conda_path = os.environ.get('CONDA_EXE', 'conda')
    script_path = os.path.abspath(sys.argv[0])
    quoted_args = ' '.join(shlex.quote(arg) for arg in sys.argv[1:])
    command = (
        f"{shlex.quote(conda_path)} run -n {shlex.quote(env_name)} "
        f"python -u {shlex.quote(script_path)}"
    )
    if quoted_args:
        command = f'{command} {quoted_args}'
    os.system(command)
    sys.exit(0)


if '--with_attribute_selection' in sys.argv:
    ensure_conda_env(get_blip_env_arg())


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def default_attribute_output_path(dataset_name, corpus_path):
    if dataset_name != 'coco_pose':
        return 'save/attribute10.json'

    corpus_name = Path(corpus_path).name
    if corpus_name == 'corpus_coco_pose_gemini.json':
        return 'save/attribute_coco_pose_gemini_blip.json'
    return 'save/attribute_coco_pose_blip.json'


def default_csv_output_path(output_attributes):
    return str(Path(output_attributes).with_suffix('.csv'))


def default_output_paths(dataset_name):
    if dataset_name == 'coco_pose':
        return {
            'labels': 'save/labels_coco_pose.npy',
            'text_labels': 'save/label_coco_pose_txt.npy',
            'train_idx': 'save/train_idx_coco_pose.npy',
            'val_idx': 'save/val_idx_coco_pose.npy',
            'attributes': None,
            'corpus': DEFAULT_COCO_POSE_CORPUS,
        }
    return {
        'labels': 'save/labels10.npy',
        'text_labels': 'label10_txt.npy',
        'train_idx': 'save/train_idx_imagenet.npy',
        'val_idx': 'save/val_idx_imagenet.npy',
        'attributes': 'save/attribute10.json',
        'corpus': 'save/corpus10.json',
    }


def collect_folder_dataset(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    print('Class to index mapping:', class_to_idx)

    all_datas = []
    labels = []
    sample_rows = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(class_dir, fname)
                labels.append(class_to_idx[class_name])
                all_datas.append(image_path)
                sample_rows.append({
                    'image_path': image_path,
                    'image_id': '',
                    'file_name': fname,
                })

    return all_datas, labels, idx_to_class, sample_rows


def has_valid_person_annotation(annotations):
    for ann in annotations:
        if ann.get('category_id') != 1:
            continue
        if ann.get('iscrowd', 0) == 1:
            continue

        bbox = ann.get('bbox', [0, 0, 0, 0])
        if bbox[2] <= 1 or bbox[3] <= 1:
            continue

        keypoints = ann.get('keypoints', [])
        if len(keypoints) != 51:
            continue
        visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        if visible > 0:
            return True
    return False


def collect_coco_pose_single_split(coco_root, split, annotation_file=None, image_dir=None):
    split_name = f'{split}2017'
    resolved_annotation_file = annotation_file or os.path.join(
        coco_root, 'annotations', f'person_keypoints_{split_name}.json'
    )
    resolved_image_dir = image_dir or os.path.join(coco_root, split_name)

    with open(resolved_annotation_file, 'r') as handle:
        coco_data = json.load(handle)

    image_lookup = {img['id']: img for img in coco_data.get('images', [])}
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    valid_image_ids = [
        image_id
        for image_id in sorted(image_lookup)
        if has_valid_person_annotation(annotations_by_image.get(image_id, []))
    ]

    all_datas = []
    labels = []
    sample_rows = []
    missing_files = 0
    for image_id in valid_image_ids:
        image_info = image_lookup.get(image_id)
        if image_info is None:
            continue
        image_path = os.path.join(resolved_image_dir, image_info['file_name'])
        if not os.path.exists(image_path):
            missing_files += 1
            continue
        all_datas.append(image_path)
        labels.append(1)
        sample_rows.append({
            'image_path': image_path,
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'split': split,
        })

    print(
        f"Loaded COCO pose split '{split}' from {resolved_annotation_file}: "
        f"{len(all_datas)} valid images"
        + (f" ({missing_files} missing files skipped)" if missing_files else '')
    )
    return all_datas, labels, {1: 'person'}, sample_rows


def collect_coco_pose_dataset(coco_root, split, annotation_file=None, image_dir=None):
    if split != 'all':
        return collect_coco_pose_single_split(
            coco_root=coco_root,
            split=split,
            annotation_file=annotation_file,
            image_dir=image_dir,
        )

    if annotation_file is not None or image_dir is not None:
        raise ValueError('--annotation_file and --image_dir are only supported with a single COCO split')

    all_datas = []
    labels = []
    label_names_dict = {1: 'person'}
    sample_rows = []
    for single_split in ('train', 'val'):
        split_datas, split_labels, _, split_rows = collect_coco_pose_single_split(
            coco_root=coco_root,
            split=single_split,
        )
        all_datas.extend(split_datas)
        labels.extend(split_labels)
        sample_rows.extend(split_rows)
    return all_datas, labels, label_names_dict, sample_rows


def write_attribute_csv(csv_path, sample_rows, attributes_dict, attr_names, dataset_name):
    if dataset_name == 'coco_pose':
        fieldnames = ['image_id', *attr_names]
    else:
        fieldnames = ['index', 'image_path', 'image_id', 'file_name', *attr_names]

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_row in enumerate(sample_rows):
            if dataset_name == 'coco_pose':
                row = {'image_id': sample_row.get('image_id', '')}
            else:
                row = {
                    'index': index,
                    'image_path': sample_row.get('image_path', ''),
                    'image_id': sample_row.get('image_id', ''),
                    'file_name': sample_row.get('file_name', ''),
                }
            for attr_name in attr_names:
                row[attr_name] = attributes_dict[attr_name]['data'][index]
            writer.writerow(row)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    type=str,
    default='folder',
    choices=['folder', 'coco_pose'],
    help='Dataset layout to read: class-folder images or COCO person keypoints',
)
parser.add_argument(
    '--root_dir',
    type=str,
    default=DEFAULT_FOLDER_ROOT,
    help='Root directory of a class-folder dataset (used when --dataset folder)',
)
parser.add_argument(
    '--coco_root',
    type=str,
    default=DEFAULT_COCO_ROOT,
    help='COCO root containing train2017/val2017 and annotations/',
)
parser.add_argument(
    '--split',
    type=str,
    default='all',
    choices=['train', 'val', 'all'],
    help='COCO split to annotate when --dataset coco_pose',
)
parser.add_argument(
    '--annotation_file',
    type=str,
    default=None,
    help='Optional explicit COCO annotation json (overrides --coco_root/--split)',
)
parser.add_argument(
    '--image_dir',
    type=str,
    default=None,
    help='Optional explicit image directory (overrides --coco_root/--split)',
)
parser.add_argument(
    '--corpus',
    type=str,
    default=None,
    help='Path to the corpus JSON that defines attributes',
)
parser.add_argument(
    '--output_labels',
    type=str,
    default=None,
    help='Path to the output label npy (auto-generated if not specified)',
)
parser.add_argument(
    '--output_attributes',
    type=str,
    default=None,
    help='Path to the output attribute JSON (auto-generated if not specified)',
)
parser.add_argument(
    '--csv_output',
    type=str,
    default=None,
    help='Path to the output attribute CSV (auto-generated from the JSON output if not specified)',
)
parser.add_argument(
    '--text_labels_output',
    type=str,
    default=None,
    help='Path to the optional text-label npy (auto-generated if not specified)',
)
parser.add_argument(
    '--train_idx_output',
    type=str,
    default=None,
    help='Path to the train-index npy (auto-generated if not specified)',
)
parser.add_argument(
    '--val_idx_output',
    type=str,
    default=None,
    help='Path to the val-index npy (auto-generated if not specified)',
)
parser.add_argument(
    '--stop_at',
    type=int,
    default=None,
    help='Process only the first N images after dataset collection',
)
parser.add_argument(
    '--with_attribute_selection',
    default=False,
    action='store_true',
    help='Enable attribute selection step (default: False)',
)
parser.add_argument(
    '--blip_env',
    type=str,
    default=DEFAULT_BLIP_ENV,
    help='Conda environment to re-launch under when BLIP attribute selection is enabled',
)
parser.add_argument(
    '--text_labels',
    action='store_true',
    default=True,
    help='If set, also save text labels to the configured text-label file',
)

args = parser.parse_args()

defaults = default_output_paths(args.dataset)
args.output_labels = args.output_labels or defaults['labels']
args.text_labels_output = args.text_labels_output or defaults['text_labels']
args.train_idx_output = args.train_idx_output or defaults['train_idx']
args.val_idx_output = args.val_idx_output or defaults['val_idx']
args.corpus = args.corpus or defaults['corpus']
args.output_attributes = args.output_attributes or default_attribute_output_path(args.dataset, args.corpus)
args.csv_output = args.csv_output or default_csv_output_path(args.output_attributes)

for path in [
    args.output_labels,
    args.output_attributes,
    args.csv_output,
    args.text_labels_output,
    args.train_idx_output,
    args.val_idx_output,
    args.corpus,
]:
    ensure_parent_dir(path)

if args.dataset == 'coco_pose':
    all_datas, labels, label_names_dict, sample_rows = collect_coco_pose_dataset(
        coco_root=args.coco_root,
        split=args.split,
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
    )
else:
    all_datas, labels, label_names_dict, sample_rows = collect_folder_dataset(args.root_dir)

if args.stop_at is not None:
    requested = max(0, args.stop_at)
    all_datas = all_datas[:requested]
    labels = labels[:requested]
    sample_rows = sample_rows[:requested]
    print(f'Will process only the first {len(all_datas)} images (--stop_at {args.stop_at})')

labels_array = np.array(labels)
np.save(args.output_labels, labels_array)
print(f'Saved {labels_array.shape[0]} labels to {args.output_labels}')

if args.text_labels:
    text_labels_array = np.array([label_names_dict[label] for label in labels], dtype=object)
    np.save(args.text_labels_output, text_labels_array)
    print(f'Saved {text_labels_array.shape[0]} text labels to {args.text_labels_output}')

classes, counts = np.unique(labels_array, return_counts=True)
print('Sample counts per class:')
for cls, cnt in zip(classes, counts):
    print(f'{cls}: {cnt}')

random.seed(42)
sample_count = min(15, labels_array.shape[0])
if sample_count > 0:
    selected_indices = random.sample(range(labels_array.shape[0]), sample_count)
    print(f'Random {sample_count} samples (idx: label):')
    for idx in sorted(selected_indices):
        print(f'{idx}: {labels_array[idx]}')
else:
    print('No samples found, skipping random label preview.')

num_samples = labels_array.shape[0]
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)
split_pt = int(0.8 * num_samples)
train_idx = indices[:split_pt]
val_idx = indices[split_pt:]
np.save(args.train_idx_output, train_idx)
np.save(args.val_idx_output, val_idx)
print(f'Saved train indices ({len(train_idx)}) to {args.train_idx_output}')
print(f'Saved val indices ({len(val_idx)}) to {args.val_idx_output}')
print('First 30 train indices:', train_idx[:30])
print('First 30 val indices:', val_idx[:30])

if args.with_attribute_selection:
    from utils.blipFeature import get_img_text_similarity, question_answering, question_answering_list

    UNCERTAIN_THRESHOLD = 0.2

    class AttributeSelection:
        def __init__(self, datas, labels, corpus, label_names_dict=None):
            self.datas = datas
            self.labels = labels
            self.label_names_dict = (
                label_names_dict
                if label_names_dict is not None
                else {label: str(label) for label in set(labels)}
            )
            self.data_length = len(self.datas)
            if isinstance(corpus, str):
                with open(corpus, 'r') as handle:
                    self.attributes_dict = json.load(handle)['Corpus']
            else:
                self.attributes_dict = copy.deepcopy(corpus)

        def match_description_to_data(self):
            for attribute in self.attributes_dict.keys():
                print('Match', attribute)
                self.attributes_dict[attribute]['data'] = [''] * self.data_length
                self.attributes_dict[attribute]['uncertain_idx'] = []
                prompts = self.attributes_dict[attribute]['prompt']
                if isinstance(prompts, str):
                    prompts = [prompts]
                    self.attributes_dict[attribute]['prompt'] = prompts
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
                    questions.append(question.replace('#LABEL', label_name))
                answer = question_answering_list(self.datas, questions)
            else:
                answer = question_answering(self.datas, question)
            self.attributes_dict[attribute]['data'] = answer

        def similarity_match(self, attribute, indexs, label=None):
            sims = []
            for prompt in self.attributes_dict[attribute]['prompt']:
                sentences = [
                    prompt.replace('#1', desc).replace('#LABEL', self.label_names_dict.get(label, ''))
                    for desc in self.attributes_dict[attribute]['word']
                ]
                sim = get_img_text_similarity(
                    [self.datas[i] for i in indexs], sentences
                ).detach().cpu().numpy()
                sims.append(sim)
            sims = np.sum(sims, axis=0)
            selected = np.argmax(sims, axis=1)
            if sims.shape[1] > 1:
                diffs = np.sort(sims, axis=1)[:, ::-1]
                uncertainty = diffs[:, 0] - diffs[:, 1]
                uncertain = np.where(uncertainty < UNCERTAIN_THRESHOLD)[0]
                self.attributes_dict[attribute]['uncertain_idx'].extend([indexs[i] for i in uncertain])
            for idx_i, img_idx in enumerate(indexs):
                self.attributes_dict[attribute]['data'][img_idx] = self.attributes_dict[attribute]['word'][selected[idx_i]]

        def match_general_description(self, attribute):
            if any('#LABEL' in prompt for prompt in self.attributes_dict[attribute]['prompt']):
                for label in set(self.labels):
                    idxs = [i for i, lbl in enumerate(self.labels) if lbl == label]
                    self.similarity_match(attribute, idxs, label)
            else:
                self.similarity_match(attribute, list(range(self.data_length)))

        def save_attributes(self, file_path):
            with open(file_path, 'w') as handle:
                json.dump(self.attributes_dict, handle, indent=4)

    attribute_selector = AttributeSelection(all_datas, labels, args.corpus, label_names_dict=label_names_dict)
    attribute_selector.match_description_to_data()
    attribute_selector.save_attributes(args.output_attributes)
    write_attribute_csv(
        args.csv_output,
        sample_rows,
        attribute_selector.attributes_dict,
        list(attribute_selector.attributes_dict.keys()),
        args.dataset,
    )
    print(f'Saved attributes to {args.output_attributes}')
    print(f'Saved attribute CSV to {args.csv_output}')
else:
    print('Skipping attribute selection (use --with_attribute_selection to enable)')
