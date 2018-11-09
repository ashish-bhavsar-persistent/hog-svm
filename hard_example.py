# -*- coding: utf-8 -*-
from skimage.transform import pyramid_gaussian
from skimage import io,color
import os 
from sklearn.externals import joblib

import dconfig
from my_common import sliding_window,my_hog


def detect_hard_example(clf, file_path, save_dir,
                     win_shape = dconfig.detector.sliding_win_shape,
                     sliding_step = dconfig.detector.sliding_step,
                     downscale = dconfig.detector.downscale,
                     decision = dconfig.detector.hard_decision):
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)    
    
    im = io.imread(file_path)
    
    print('detect_hard_example...')
    hard_example_counter = 0
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        if im_scaled.shape[0] < win_shape[1] \
            or im_scaled.shape[1] < win_shape[0]:
            print('out of boundary')
            break

        for (x, y, im_window) in sliding_window(im_scaled, win_shape, sliding_step):
            if im_window.shape[0] != win_shape[1] or im_window.shape[1] != win_shape[0]:
                continue
            
            if dconfig.hog.multichannel == False:
                im_window = color.rgb2gray(im_window)
            
            fd = my_hog(im_window)
            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:
                if clf.decision_function(fd) > decision:
                    name = "%s/%s_%d%s" % (save_dir, filename, 
                                           hard_example_counter, extension)
                    io.imsave(name, im_window)
                    hard_example_counter += 1
    print('{} hard_example found from {}'.format(hard_example_counter, filename))


def process_im_folder(im_dir, save_dir=None):
    '''
    loop each image in $im_dir
    crop pric with high confidence and save them into $save_dir
    '''
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    print('loading model...')      
    clf = joblib.load(dconfig.svm.model)
    print('loaded model')  
    
    file_list = os.listdir(im_dir)
    for f in file_list:
        (filename, extension) = os.path.splitext(f)
        if extension == '.png' or \
            extension == '.jpg' or \
            extension == '.pgm':
            file_path = os.path.join(im_dir, f)
            if os.path.isfile(file_path):
                detect_hard_example(clf, file_path, save_dir)

if __name__ == '__main__':
    print('loading model...')      
    clf = joblib.load(dconfig.svm.model)
    print('loaded model')  
    detect_hard_example(clf, 'test_image/test_neg/00001169.png',
                        save_dir='tmp/')