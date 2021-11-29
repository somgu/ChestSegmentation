import os
import natsort
import pydicom
from PIL import Image
import numpy as np
import glob

class Dataset:
    def __init__(self, dataset_path):
        self.dcm_path_list = []  # dcm 파일명 리스트
        self.mask_path_list = []  # gruond truth 파일명 리스트
        self.dicoms = []  # dcm 파일(이미지 + desc) 리스트
        self.masks = []  # ground truth 파일(이미지) 리스트

        # dcm, ground truth 이미지 파일명 읽어오기. 읽어오면서 파일을 정렬
        data_list = natsort.natsorted(os.listdir(dataset_path))
        self.dcm_path_list = glob.glob(os.path.join(dataset_path, '*.dcm'))
        self.mask_path_list = glob.glob(os.path.join(dataset_path, '*.png'))

        for dcm in self.dcm_path_list:
            self.dicoms.append(pydicom.dcmread(dcm))

        for mask in self.mask_path_list:
            self.masks.append(np.array(Image.open(mask)))
