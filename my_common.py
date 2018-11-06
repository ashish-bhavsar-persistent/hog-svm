# -*- coding: utf-8 -*-
import numpy as np 
from skimage.feature import hog


class dconfig:
    class dataset:
        width = 64
        height = 128
        #width = 48
        #height = 96
        pos_fold = ''
        neg_fold = ''
    class svm:
        #kernel = 'linear'
        #C = 1000
        kernel = 'rbf'
        C = 100
        #gamma = 0.001
        model = 'svm.model'
    class hog:
        #pixels_per_cell = (8, 8)
        #cells_per_block = (4, 4)
        pixels_per_cell = (6, 6)
        cells_per_block = (2, 2)
        block_norm='L2-Hys'
        multichannel=None
        #multichannel=False
    class detector:
        downscale = 1.25
        sliding_win_shape = (64, 128)
        #sliding_step = (10, 10)
        #sliding_win_shape = (48, 96)
        sliding_step = (6, 6)
        decision = 2.0   #0.5
        nms_overlap = 0.3 #0.3
    

def sliding_window(image, window_size, step_size):
    '''    
    yxd: 
        this sliding window is not perfect
        at the most left, and down, some object will missed
    
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


def non_max_suppression2(boxes, probs=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
        # yxd
		#overlap = (w * h) / (area[idxs[:last]] + area[i])
        #overlap2 = (w * h) / area[i]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
    
	return boxes[pick].astype("int"), pick


def my_hog(image):
    return hog(image,orientations=9,
        pixels_per_cell=dconfig.hog.pixels_per_cell,  # 6
        cells_per_block=dconfig.hog.cells_per_block,  # 2
        block_norm=dconfig.hog.block_norm,
        visualize=False,
        visualise=None,
        transform_sqrt=False,
        feature_vector=True,
        multichannel=dconfig.hog.multichannel) ##### None


