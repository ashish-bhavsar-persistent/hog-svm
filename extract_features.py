from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from my_common import dconfig, my_hog

if dconfig.hog.multichannel == False:
    ismultichannel = False
else:
    ismultichannel = True

def _hog_dir(im_dir, feat_dir):
    if not os.path.isdir(im_dir):
        print('error image dir:' + im_dir)
        return

    if not os.path.isdir(feat_dir):
        os.makedirs(feat_dir)
                
    print("hog %s, save to %s ..." % (im_dir, feat_dir))
    for im_item in glob.glob(os.path.join(im_dir, "*")): 
        if ismultichannel:
            im_arr = imread(im_item)
        else:
            im_arr = imread(im_item, as_gray=True)
        fd = my_hog(im_arr)
        fd_name = os.path.split(im_item)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(feat_dir, fd_name)
        joblib.dump(fd, fd_path)
    print('done')


def t0_hog():
    im_dir = "E:/0workspace/python/ml/svmtest/data/images/neg_person" 
    ft_dir = "E:/0workspace/python/ml/hog_svm/feat/t0_hog/neg"
    _hog_dir(im_dir, ft_dir) 

    im_dir = "E:/0workspace/python/ml/svmtest/data/images/pos_person"
    ft_dir = "E:/0workspace/python/ml/hog_svm/feat/t0_hog/pos"
    _hog_dir(im_dir, ft_dir) 
    
        
if __name__=='__main__':
    t0_hog()
    print('exit')
