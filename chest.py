from keras_unet.models import custom_unet
from keras_unet.utils import plot_imgs
from Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

dataset = Dataset('./ChestCT_GT')
dcm_path_list = dataset.dcm_path_list
gt_path_list = dataset.gt_path_list
dcm_list = dataset.dicoms
gt_list = dataset.gt



