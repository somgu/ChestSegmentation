import os
import natsort
import pydicom
from PIL import Image
import numpy as np

class Dataset:
    def __init__(self, dataset_path):
        self.dcm_list = []  # dcm 파일명 리스트
        self.gt_list = []  # gruond truth 파일명 리스트
        self.dicoms = [] # dcm 파일(이미지 + desc) 리스트
        self.gt = [] # ground truth 파일(이미지) 리스트

        # dcm, ground truth 이미지 파일명 읽어오기. 읽어오면서 파일을 정렬
        data_list = natsort.natsorted(os.listdir(dataset_path))
        data_list = data_list[0::2]
        for data in data_list:
            file_name = data.split('\\')[-1]
            self.dcm_list.append(file_name + '.dcm')
            self.gt_list.append(file_name + '.png')

        for dcm in self.dcm_list:
            self.dicoms.append(pydicom.dcmread(dcm))

        for ground_truth in self.gt_list:
            self.gt.append(np.array(Image.open(ground_truth)))
