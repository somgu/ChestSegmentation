from Dataset import Dataset
import Augmentation
import random


origin_dataset = Dataset("ChestCT_GT/ChestCT_DCM", "ChestCT_GT/ChestCT_GT_Label")
augmentation = Augmentation(origin_dataset.dcm_list, origin_dataset.gt_list)
rotated_dcm, rotated_gt = augmentation.rotation(random.randrange(1, 180))
rotated_dcm = augmentation.write_dicom("ChestCT_GT/Augmented_ChestCT_DCM", rotated_dcm)
rotated_gt = augmentation.write_gt("ChestCT_GT/Augmented_ChestCT_GT_Label", rotated_gt)
augmentation.show_dicom_in_plt(rotated_dcm[0])
augmentation.show_gt_in_plt(rotated_gt[0])
