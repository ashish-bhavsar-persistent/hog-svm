# -*- coding: utf-8 -*-
import numpy as np 
from skimage.transform import pyramid_gaussian
from skimage import io,color
from sklearn.externals import joblib
import cv2
import matplotlib.pyplot as plt 
import os 

import dconfig
from my_common import sliding_window,non_max_suppression2,my_hog


def draw_box(image, confidence, startX, startY, endX, endY,
             rectColor=(0, 255, 0),
             textColor=(0, 255, 0)):
    #print("draw_box, x:{},y:{},w:{},h:{},sc:{:.2f}".format(
    #        startX,startY,endX-startX,endY-startY, confidence))

    cv2.rectangle(image, (startX, startY), (endX, endY), rectColor, 2)

    if confidence > 0:
        # display the prediction
        label = "  {:.1f}".format(confidence * 100)
        #print("[INFO] {}".format(label))
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2)


def hog_svm_predict(clf, im,
                     win_shape = dconfig.detector.sliding_win_shape,
                     sliding_step = dconfig.detector.sliding_step,
                     downscale = dconfig.detector.downscale,
                     decision = dconfig.detector.decision):
    detections = []
    scale = 0
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        if im_scaled.shape[0] < win_shape[1] \
            or im_scaled.shape[1] < win_shape[0]:
            print('out of boundary, scale:{}'.format(scale) )
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
                #dd = clf.decision_function(fd)
                #yxd, hack .....
                dd = clf.decision_function(fd) * (downscale**scale)
                if dd > decision:
                    detections.append((int(x * (downscale**scale)),
                                       int(y * (downscale**scale)),  
                                       int(win_shape[0] * (downscale**scale)),
                                       int(win_shape[1] * (downscale**scale)),
                                       dd,
                                       downscale**scale))

        scale += 1
        
    return detections


def _detect_boxes(file_path, clf, draw_detects = False):
    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)    
    
    im = io.imread(file_path)
    #print('loading model...')      
    #clf = joblib.load(dconfig.svm.model)
    #print('loaded model')      

    #print('predict... :{}'.format(file_path))
    detections = hog_svm_predict(clf, im)
    if len(detections) <= 0:
        print('no found in {}'.format(file_path))
        return None, None, None

    if draw_detects:
        clone = im.copy()
        for (x, y, w, h, _, _) in detections:
            draw_box(clone, 0, x, y, x + w, y + h)
        
        plt.axis("off")
        plt.imshow(clone)
        plt.title("Raw Detection before NMS")
        plt.show()            
            
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h, _, _,) in detections])
    probs = [prob[0] for (x, y,  w, h, prob, _) in detections]
    probs = np.array(probs)
    boxes, picks = non_max_suppression2(rects, probs = probs, overlapThresh = dconfig.detector.nms_overlap)

    detects = np.array(detections)
    
    return im, detects, picks

def draw_boxes(file_path, clf):
    im, detects, picks = _detect_boxes(file_path, clf, draw_detects=True)
    if im is None:
        return

    clone = im.copy()
    for ((x, y, w, h, prob, ss), i) in zip(detects[picks], range(len(picks))):
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        print("draw_box, x:{},y:{},w:{},h:{},prob:{:.2f},scale:{:.2f}".format(
            x, y, w, h, prob, ss))
        draw_box(clone, prob, x, y, x+w, y+h)

    plt.axis("off")
    plt.imshow(clone)
    plt.title("Final Detections after applying NMS")
    plt.show()


def detect_and_save_boxes(file_path, save_dir):    
    im, detects, picks = _detect_boxes(file_path, draw_detects=True)
    if im is None:
        return
    
    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)
    
    for ((x, y, w, h, prob, ss), i) in zip(detects[picks], range(len(picks))):
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    
        name = "%s/%s_%d%s" % (save_dir, filename, i, extension)
        #print('save{}:{}'.format(i, name))
        #TODO: find a better way to do the rescale .... 
        #FIXME:            
        scale = 0
        if ss > 1:
            print('scale note:{}'.format(name))
            for im_scaled in pyramid_gaussian(im, max_layer = 1, downscale = ss):
                if scale == 1:
                    x = int(x/ss)
                    y = int(y/ss)
                    w = int(w/ss)
                    h = int(h/ss)
                    io.imsave(name, im_scaled[y:y+h, x:x+w])
                scale += 1
        else:
            io.imsave(name, im[y:y+h, x:x+w])
    

def process_im_folder(im_dir, save_dir=None, show_pic=False):
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
                draw_boxes(file_path, clf)


if __name__ == '__main__':
    #process_im_file('test_neg/00001169.png', 'fake/neg/', show_pic=True)
    #process_im_folder('E:/0workspace/python/ml/svmtest/INRIAPerson/Train/neg', 'fake/neg/')
    
    process_im_folder('test_image/test1', show_pic=True)

    