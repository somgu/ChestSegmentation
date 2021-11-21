import os
import natsort
import pydicom

class Dataset:
    def __init__(self, dcm_path, gt_path):
        self.dcm_list = []  # dcm 파일을 가지고 있는 멤버 변수
        self.gt_list = []  # gruond truth 파일을 가지고 있는 멤버 변수
        self.dcm_folder_path = dcm_path
        self.gt_folder_path = gt_path
        self.dicoms = []
        self.ground_truth = []

        # dcm, ground truth 이미지 파일명 읽어오기. 읽어오면서 파일을 정렬
        dcm_file_list = os.listdir(dcm_path)
        gt_file_list = os.listdir(gt_path)

        dcm_num_list = sorted([int(dcm.split('_')[1].split('_')[0]) for dcm in dcm_file_list])
        gt_num_list = sorted([int(gt.split('_')[1].split('_')[0]) for gt in gt_file_list])

        self.dcm_list = natsort.natsorted(dcm_file_list)
        self.gt_list = natsort.natsorted(gt_file_list)

        for idx, dcm in enumerate(self.dcm_list):
            self.dcm_list[idx] = os.path.join(dcm_path, dcm)

        for idx, gt in enumerate(self.gt_list):
            self.gt_list[idx] = os.path.join(gt_path, gt)

        print(self.dcm_list)
        print(self.gt_list)

    def get_file_list(self):
        dcm_path = []
        for dcm in self.dcm_list:
            dcm_path.append(os.path.join(self.dcm_folder_path, dcm))

        gt_path = []
        for gt in self.gt_list:
            gt_path.append(os.path.join(self.gt_folder_path, gt))

        return dcm_path, gt_path

    def get_dicoms(self):
        for idx, dcm in enumerate(self.dcm_list):
            self.dicoms.append(pydicom.filewriter.dcmwrite())