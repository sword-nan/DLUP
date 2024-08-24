import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np

from model.model import Model
from model.model_utils import set_seed, get_device, ModelTrainer, ModelTester
from data_utils.dataset import MzmlDataset
from data_utils.dataloader import Loader
from utils.io import create_dir
from utils.transform import transform_time

set_seed(42)

root = '/data/xp/train_test_data/astral_20231016_300ngPlasmaSample'
save_root = '/data/xp/result/astral_20231016_300ngPlasmaSample'
model_root = '/data/xp/model/astral_20231016_300ngPlasmaSample'

create_dir(save_root)
create_dir(model_root)

path_configs = {
    'root': root,
    'save_root': save_root,
    'train_data': os.path.join(root, 'train/identification/penalty_MAE_peaksum/collection.npy'),
    'test_data': os.path.join(root, 'test/identification/penalty_MAE_peaksum/collection.npy'),
    'train_save_path': os.path.join(save_root, 'train_identification.npy'),
    'test_save_path': os.path.join(save_root, 'test_identification.npy')
}

model_configs = {
    'd_in': 6,
    'num_layers': 4,
    'feedward_dim': 128,
    'n_head': 2,
    'dropout': 0.3,
    'd_out': 1
}

trainer_configs = {
    'save_path': os.path.join(model_root, 'identification.ckpt'),
    'n_epochs': 1000,
    'scheduler':{
        "warmup_steps": 25000,
        "total_steps": 500000,
    }
}

device = get_device()

loader = Loader(
    0.1,
    MzmlDataset,
    batch_size=8192,
    n_works=8
)


def custom_metric_fn(input: torch.Tensor, target: torch.Tensor):
    temp = torch.zeros_like(input)
    temp[input < 0.5] = 0
    temp[input >= 0.5] = 1
    return -(temp == target).float().sum()

start = time.time()
train_data, validation_data = loader.train_loader(path_configs['train_data'])
end = time.time()
hour, minute, second = transform_time(end - start)
print('读取数据完毕! 耗时 {:.2f}时 {:.2f}分 {:.2f}秒'.format(hour, minute, second))

model = Model(
    **model_configs
)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss(reduction='sum')

trainer = ModelTrainer(
    model,
    train_data,
    validation_data,
    optimizer,
    criterion,
    device=device,
    train_configs=trainer_configs,
    custom_metrics_name='accuracy',
    custom_metrics_fn=custom_metric_fn
)

loss_metrics, custom_metrics = trainer.train()

np.save('./loss_metrics.npy', loss_metrics)
np.save('./custom_metrics.npy', custom_metrics)

model = Model(
    **model_configs
)

model_state = torch.load(trainer_configs['save_path'])
model.load_state_dict(model_state)
criterion = nn.BCELoss(reduction='sum')
model = model.to(device)

test_data, test_info = loader.test_loader(path_configs['test_data'])
train_data, train_info = loader.test_loader(path_configs['train_data'])

tester = ModelTester(
    model,
    test_data,
    criterion,
    device
)

def peptide_identification(precursor_identification, library):
    identification = defaultdict(set)
    for f, precursors in precursor_identification.items():
        for precursor in precursors:
            identification[f].add(library[precursor]['StrippedSequence'])
    return identification


def split_score_by_file(metadata):
    dtype = np.dtype([('score', np.float32), ('flag', np.bool_), ('info', object)])
    info = metadata['info']
    score_split_by_file = defaultdict(list)

    for score, info in zip(metadata['score'], metadata['info']):
        file = info[0]
        flag = True
        if 'DECOY-' in info[2][0]:
            flag = False

        score_split_by_file[file].append(
            np.array([(
                score, 
                flag, 
                (info[1], info[2])
            )],
            dtype=dtype
        ))

    for key in score_split_by_file.keys():
        score_split_by_file[key] = np.concatenate(score_split_by_file[key], axis=0)

    score_split_by_file = dict(score_split_by_file)

    return score_split_by_file

# test data
tester.replace_dataloader(test_data)
test_score = tester.test(test_info)
test_score = split_score_by_file(test_score)
np.save(path_configs['test_save_path'], test_score)

# train data
tester.replace_dataloader(train_data)
train_score = tester.test(train_info)
train_score = split_score_by_file(train_score)
np.save(path_configs['train_save_path'], train_score)


def calculate_score_fdr_cutoff(scores, flag):
    indices = np.argsort(scores)[::-1]
    scores = scores[indices]
    flag = flag[indices]
    numDecoys = 0
    count = 0
    for i in range(len(scores)):
        if not flag[i]: numDecoys += 1
        curFDR = numDecoys/(count+1)
        if curFDR > 0.01:
            if count < 1/0.01: return -1
            return scores[i-1]
        count += 1

    return scores[-1]

def fdr(score_split_by_file):
    precursor_identification = defaultdict(set)
    for f, metadata in score_split_by_file.items():
        score, flag, info = metadata['score'], metadata['flag'], metadata['info']
        threshold = calculate_score_fdr_cutoff(metadata['score'], metadata['flag'])
        indices = (score >= threshold) & flag
        for item in info[indices]:
            precursor_identification[f].add(item[1])
        print(f, threshold, len(precursor_identification[f]))
    return precursor_identification

precursor_result = fdr(test_score)
sn_train = np.load('/data/xp/label/astral_20231016_300ngPlasmaSample/train_identification.npy', allow_pickle=True).item()
sn_test = np.load('/data/xp/label/astral_20231016_300ngPlasmaSample/test_identification.npy', allow_pickle=True).item()

for f, val in sn_train.items():
    print(f, len(val), len(precursor_result[f]))

for f, val in sn_test.items():
    print(f, len(val), len(precursor_result[f]))