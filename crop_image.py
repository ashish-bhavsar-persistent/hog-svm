# -*- coding: utf-8 -*-

import os
import cv2
from skimage import io,transform,color
from skimage.transform import pyramid_gaussian
import random
#import matplotlib.pyplot as plt 


def sliding_window(image, window_size, step_size):
    '''
    return a bunch of images:
        sliding_window(im, (64,128), 10)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])
            
def random_window(image, window_size, num):
    '''
    return 10 random images:
        sliding_window(im, (64,128), 10)
    '''
    x_max = image.shape[1]
    y_max = image.shape[0]
    cx_max = window_size[0]
    cy_max = window_size[1]
    if cx_max >= x_max or cy_max >= y_max:
        print('image is too small to crop')
        return
    
    for i in range(num):
        x = random.randint(0, x_max-cx_max-1)
        y = random.randint(0, y_max-cy_max-1)
        #print('(%d, %d, %d, %d)' % (x, y, x + cx_max, y + cy_max))
        yield (x, y, image[y: y + cy_max, x: x + cx_max])
            
def create_crop_images(file_path='648x432.jpg', crop_sz=(64, 128), save_dir='./'):
    im = cv2.imread(file_path)
    if im is None:
        return

    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)

    # auto determine how many im to create
    crop_num = int(im.shape[0] * im.shape[1] / (crop_sz[0] * crop_sz[1]))
    crop_count = crop_num
    

    for im_scaled in pyramid_gaussian(im, downscale = 1.25, multichannel=True):
        if im_scaled.shape[0] < crop_sz[1] or im_scaled.shape[1] < crop_sz[0]:
            #print('kkkk', im_scaled.shape)
            break
 
        if crop_count <= 0:
            return
        
        for (_, _, im_croped) in random_window(im_scaled, crop_sz, int(crop_count / 2)):
            im_name = '%s_%s.png' % (filename, crop_count)
            im_path = '%s/%s' % (save_dir, im_name)
            #print(im_name, im_croped.shape)
            #TODO: do the color space transfer
            io.imsave(im_path, im_croped)
            #io.imsave(im_name, cv2.cvtColor(im_croped, cv2.COLOR_BGR2RGB))
            crop_count = crop_count - 1
            #io.imshow(im_croped)
            #plt.imshow(cv2.cvtColor(im_croped, cv2.COLOR_BGR2RGB))


def crop_im_folder(im_dir, save_dir):
    '''
    im_dir = "E:/0workspace/python/ml/svmtest/INRIAPerson/Train/neg"
    save_dir="E:/0workspace/python/ml/hog_svm/feat/neg_k_im"
    #save_dir="E:\\0workspace\\python\\ml\\hog_svm\\feat\\neg_k'
    crop_im_folder(im_dir=im_dir, save_dir=save_dir)
    '''
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    file_list = os.listdir(im_dir)
    for f in file_list:
        (filename, extension) = os.path.splitext(f)
        if extension == '.png' or \
            extension == '.jpg' or \
            extension == '.pgm':
            file_path = os.path.join(im_dir, f)
            if os.path.isfile(file_path):
                create_crop_images(file_path=file_path,
                                   crop_sz=(64, 128),
                                   save_dir=save_dir)
 
def scale_im_folder(im_dir, save_dir, im_sz=(96, 48), name_sufix='jlq.png'):
    '''
    im_dir = "E:/0workspace/python/ml/svmtest/INRIAPerson/70X134H96/Test/pos"
    save_dir="E:/0workspace/python/ml/hog_svm/feat/pos_k_im" 
    scale_im_folder(im_dir=im_dir, save_dir=save_dir)
    '''
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    file_list = os.listdir(im_dir)
    for f in file_list:
        (filename, extension) = os.path.splitext(f)
        if extension == '.png' or \
            extension == '.jpg' or \
            extension == '.pgm':
            #file_path = os.path.join(im_dir, f)
            file_path = "%s/%s" % (im_dir, f)
            #print(file_path)
         
            if os.path.isfile(file_path):
                im = io.imread(file_path)
                if im is not None:
                    #cv2.resize(im, (64,128))
                    r_im = transform.resize(im, im_sz)
                    if len(r_im.shape) > 2 :
                        r_im = color.rgb2gray(r_im)
                    
                    s_file_path = "%s/%s_%s" % (save_dir, filename, name_sufix)
                    #print(s_file_path)
                    #io.imsave(s_file_path, r_im[:,:,0:3])
                    io.imsave(s_file_path, r_im)
               
if __name__ == '__main__':
    #create_crop_images()
    
    #############crop########################
    #im_dir = "E:/0workspace/python/ml/svmtest/INRIAPerson/Train/neg"
    #save_dir="E:/0workspace/python/ml/hog_svm/feat/neg_k_im"
    
    #crop_im_folder(im_dir=im_dir, save_dir=save_dir)


    #############resize########################
    #im_dir = "E:/0workspace/python/ml/svmtest/INRIAPerson/70X134H96/Test/pos"
    #im_dir = "E:/0workspace/python/ml/svmtest/INRIAPerson/96X160H96/Train/pos"
    #im_dir = "E:/0workspace/python/ml/svmtest/DaimlerBenchmark/Data/TrainingData/Pedestrians/48x96"
    #save_dir="E:/0workspace/python/ml/hog_svm/feat/pos_g_im" 

    im_dir = "E:/0workspace/python/ml/hog_svm/feat/neg_k_im"
    save_dir="E:/0workspace/python/ml/hog_svm/feat/neg_g_im" 
    scale_im_folder(im_dir=im_dir, save_dir=save_dir)

    
    