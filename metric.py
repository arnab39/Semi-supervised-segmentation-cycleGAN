### Implementation of Intersection Over Union metric for evaluating results

import numpy as np
from PIL import Image


def metric(real_seg, test_seg):
    '''
    Parameters
    ----------

    Both must be of the type PIL.Image

    real_seg : The real segmentation provided as ground truth
    test_seg : The segmentation results obtained from the network

    Returns
    -------
    Returns a dict containing IoU between the images along with the extra color labels that are in neither
    '''
    real_seg_array = np.array(real_seg)
    test_seg_array = np.array(test_seg)

    #### The class labels in real and segmented images
    real_labels = np.unique(real_seg_array)
    test_labels = np.unique(test_seg_array)

    #### For the labels commmon between the two arrays
    common_labels = np.intersect1d(real_labels, test_labels)

    #### The labels that are extra in the test and real case
    extra_labels_in_real_seg = [i for i in real_labels if i not in common_labels]
    extra_labels_in_test_seg = [i for i in test_labels if i not in common_labels]

    IoU = {}

    for i in range(common_labels.shape[0]):
        common = 0
        test_not_real = 0
        real_not_test = 0

        for j in range(real_seg_array.shape[0]):
            for k in range(test_seg_array.shape[0]):
                if(real_seg_array[j][k] == common_labels[i]):
                    if(test_seg_array[j][k] == common_labels[i]):
                        common += 1
                    else:
                        real_not_test += 1
                elif(test_seg_array[j][k] == common_labels[i]):
                    test_not_real += 1
        
        ratio = (common)/(common + test_not_real + real_not_test + 1e-6)

        IoU[common_labels[i]] = ratio
    
    IoU['extra_labels_in_real_seg'] = extra_labels_in_real_seg
    IoU['extra_labels_in_test_seg'] = extra_labels_in_test_seg

    return IoU
