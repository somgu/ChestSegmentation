# chest dataset을 train, val, test로 나눔

import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

dcm_files = glob("./ChestCT_GT/*.dcm")
dcms = [name.replace('.dcm', '') for name in dcm_files]

train_names, test_names = train_test_split(dcms, test_size=0.2, random_state=42, shuffle=True)
val_name, test_names = train_test_split(test_names, test_size=0.5, random_state=42, shuffle=True)

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        dcm = file.split('\\')[-1] + '.dcm'
        gt = file.split('\\')[-1] + '.png'
        shutil.copy(os.path.join(source_path, dcm), destination_path)
        shutil.copy(os.path.join(source_path, gt), destination_path)
    return

source_dir = './ChestCT_GT'

train_dir = './Dataset/train'
test_dir = './Dataset/test'
val_dir = './Dataset/val'
batch_move_files(train_names, source_dir, train_dir)
batch_move_files(test_names, source_dir, test_dir)
batch_move_files(val_name, source_dir, val_dir)