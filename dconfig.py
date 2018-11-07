# -*- coding: utf-8 -*-

class dataset:
    #width = 64
    #height = 128
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
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)
    #pixels_per_cell = (6, 6)
    #cells_per_block = (2, 2)
    block_norm='L2-Hys'
    multichannel=None
    #multichannel=False
    
    
class detector:
    downscale = 1.25
    sliding_win_shape = (64, 128)
    #sliding_step = (10, 10)
    #sliding_win_shape = (48, 96)
    sliding_step = (5, 5)
    #sliding_step = (10, 10)
    decision = 0.8   #0.5
    nms_overlap = 0.5 #0.3


svm.model='feat/8.3/48x96/svm.model'
detector.sliding_win_shape = (48,96)
detector.decision = 3.0