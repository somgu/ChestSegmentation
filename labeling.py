from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

connectivity_4 = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]])

connectivity_8 = np.array([[1, 1, 1],
                           [1, 0, 0],
                           [0, 0, 0]])


def find_labels(labels, r, c, neighbors):
    tmp_labels = labels[r - 1:r + 2, c - 1:c + 2] * neighbors
    return np.sort(tmp_labels[np.nonzero(tmp_labels)])


def find_segmentation(bin_img, val):
    labels = np.zeros_like(bin_img, dtype='uint8')
    labels.fill(val)

    for r, row in enumerate(bin_img):
        for c, pixel in enumerate(row):
            if(val != pixel):
                labels[r,c] = 0

    return labels



def connected_component_labeling(bin_img, connectivity=connectivity_8):
    equiv = []
    labels = np.zeros_like(bin_img, dtype='uint8')
    next_label = 1

    for r, row in enumerate(bin_img):
        for c, pixel in enumerate(row):
            if pixel != 0:
                neighbors = bin_img[r - 1:r + 2, c - 1:c + 2] * connectivity
                num_neighbors = np.count_nonzero(neighbors)

                if num_neighbors == 0:
                    labels[r, c] = next_label
                    equiv.append([next_label, next_label])
                    next_label += 1
                else:
                    L = find_labels(labels, r, c, neighbors)
                    try:
                        labels[r, c] = np.min(L)
                    except ValueError:
                        pass

                    uni_L = np.unique(L)
                    if len(uni_L) > 1:
                        for i, e in enumerate(equiv):
                            if uni_L[0] in e:
                                equiv[i].extend(uni_L[1:])
                                equiv[i] = list(sorted(set(equiv[i])))

    for e in equiv:
        for f in reversed(e):
            labels[labels == f] = e[0]

    return labels


def threshold_labels(labels, threshold=1):
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    thr_elements = unique_elements[counts_elements > threshold]

    thr_labels = np.zeros_like(labels)

    cnt = 0
    for e in thr_elements:
        if e != 0:
            cnt += 1
            thr_labels[labels == e] = cnt
    return thr_labels
