import cv2
import random
import albumentations
import imageio
import imgaug
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import natsort
import pydicom
import Image
import matplotlib.pyplot as plt
from pathlib import Path

origin_dcm_path = r"C:\Users\user\Desktop\Project\ChestCT_GT\ChestCT_DCM"  # dcm 원본 파일 디렉터리 위치
origin_gt_path = r"C:\Users\user\Desktop\Project\ChestCT_GT\ChestCT_GT_Label"  # gt 원본 파일 디렉터리 위치
dcm_file_list = os.listdir(origin_dcm_path)  # dcm 파일 이름 리스트
gt_file_list = os.listdir(origin_gt_path)  # dcm 파일 이름 리스트
dcm_num_list = sorted([int(dcm.split('_')[1].split('_')[0]) for dcm in dcm_file_list])
gt_num_list = sorted([int(gt.split('_')[1].split('_')[0]) for gt in gt_file_list])
dcm_file_list = natsort.natsorted(dcm_file_list)  # 순서대로 정렬된 dcm 파일 이름 리스트
gt_file_list = natsort.natsorted(gt_file_list)  # 순서대로 정렬된 gt 파일 이름 리스트
dcm_path_list = []
gt_path_list = []
for idx, (dcm, gt) in enumerate(zip(dcm_file_list, gt_file_list)):
    dcm_path_list.append(os.path.join(origin_dcm_path, dcm))  # "ChestCT_GT/ChestCT_DCM/Chest_1_100000059.dcm"
    gt_path_list.append(os.path.join(origin_gt_path, gt))  # "ChestCT_GT/ChestCT_DCM/Chest_1_100000059.gt"
print(dcm_path_list, gt_path_list)

# 이미지 변형이 가능한 dcm, gt 리스트
dcm_list = []
dcm_pixel_array_list = []
gt_list = []
segmentation_maps_list = []
for idx, dcm in enumerate(dcm_path_list):
    dcm_list.append(pydicom.dcmread(dcm))
    dcm_pixel_array_list.append(dcm_list[idx].pixel_array)
for idx, gt in enumerate(gt_path_list):
    gt_list.append(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
for idx, gt in enumerate(gt_list):
    segmentation_maps_list.append(imgaug.SegmentationMapsOnImage(gt, shape=gt.shape))
# augmentation functions
# augmented_dcm = dcm_list
# augmented_gt = gt_list
print(gt_list[0])
print(segmentation_maps_list[0])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# dcm 정보
dcm_min = dcm_list[0].pixel_array.min
dcm_max = dcm_list[0].pixel_array.max
dcm_type = dcm_list[0].pixel_array.dtype
dcm_shape = dcm_list[0].pixel_array.shape
dcm_size = dcm_list[0].pixel_array.size
# ground truth  정보
gt_min = gt_list[0].min
gt_max = gt_list[0].max
gt_type = gt_list[0].dtype
gt_shape = gt_list[0].shape
gt_size = gt_list[0].size

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # 50%의 이미지를 좌우반전 시킴
        iaa.Flipud(0.2),  # 20%의 이미지를 상하반전 시킴
        # 이미지 크롭 & 패딩
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 512)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 각 축에 대해 80%~120% 만큼 스케일링
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 각 축에 대해 -20% ~ 20% 만큼 이동
            rotate=(-45, 45),  # -45 ~ 45도 회전
            shear=(-16, 16),  # -16, 16 도 깍기..?
            order=[0, 1],  # 최근접보간 또는 쌍선형보간 수행
            cval=(0, 512),
            mode=ia.ALL
        )),
        iaa.SomeOf((0, 5),
                   [
                       sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # 시그마 0 ~ 3.0 사이에서 블러 처리
                           iaa.AverageBlur(k=(2, 7)),
                           # iaa.MedianBlur(k=(3, 11)),
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                       # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 512), per_channel=0.5),
                       # iaa.OneOf([
                       #     iaa.Dropout((0.01, 0.1), per_channel=0.5),
                       #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       # ]),
                       # iaa.Invert(0.05, per_channel=True),
                       # iaa.Add((-10, 10), per_channel=0.5),
                       # iaa.OneOf([
                       #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       # ]),
                       # iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                       # iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

# augmentation for loop
AUGMENTATION_BATCH_SIZE = 50
for idx in range(AUGMENTATION_BATCH_SIZE):
    dcm, gt = seq(images=dcm_pixel_array_list, segmentation_maps=segmentation_maps_list)
    temp = dcm_list  # dcm 파일 형식을 맞추기 위해 dcm_list 원본을 카피
    for jdx, (aug_dcm, aug_gt) in enumerate(zip(dcm, gt)):
        temp[jdx].PixelData = np.ascontiguousarray(aug_dcm, dtype=np.int16)
        pydicom.filewriter.dcmwrite(  # Chest_1_100000059_1.dcm
            os.path.join(
                r"C:\Users\user\Desktop\Project\ChestCT_GT\Augmented_ChestCT_DCM",
                Path(dcm_file_list[jdx]).stem + "_" + str(idx + 1) + ".dcm"
            ),
            temp[jdx]
        )
        aug_gt = np.ascontiguousarray(aug_gt.get_arr(), dtype=np.uint8)
        imageio.imwrite(
            os.path.join(
                r"C:\Users\user\Desktop\Project\ChestCT_GT\Augmented_ChestCT_GT_Label",
                Path(gt_file_list[jdx]).stem + "_" + str(idx + 1) + ".png"
            ),
            aug_gt
        )
