import matplotlib.pyplot as plt
import matplotlib.image as image
import cv2
import os
import pydicom

'''
import matplotlib.pyplot as plt
import matplotlib.image as image
import pydicom

dcm = pydicom.dcmread(r"")
plt.imshow(dcm.pixel_array, cmap=plt.cm.gray)
plt.show()

gt = image.imread(r"")
plt.imshow(gt, cmap=plt.cm.gray)
plt.show()
'''

def show_dicom_in_plt(dcm):
    pydicom.dcmread(dcm)
    plt.imshow(dcm.pixel_array, cmap=plt.cm.gray)
    plt.show()


def show_dicom_in_cv2(dcm):
    cv2.imshow('dcm', dcm.pixel_array)
    cv2.waitKey(0)
    cv2.destroyWindow('dcm')


def show_gt_in_plt(gt):
    gt_image = image.imread(gt)
    plt.imshow(gt_image, cmap=plt.cm.gray)
    plt.show()


def show_gt_in_cv2(gt):
    cv2.imshow('ground truth', gt)
    cv2.waitKey(0)
    cv2.destroyWindow('ground truth')


def write_dicom(folder_path, dcm):
    pydicom.filewriter.dcmwrite(os.path.join(folder_path, dcm))
    print("dicom saving completed")


def write_gt(folder_path, gt):
    cv2.imwrite(os.path.join(folder_path, gt))
    print("ground truth saving completed")
