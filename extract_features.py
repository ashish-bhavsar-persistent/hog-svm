from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
import dconfig
from my_common import my_hog

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


def test(imdir, ftdir):
    im_dir = "{}/pos".format(imdir)
    ft_dir = "{}/pos".format(ftdir)
    _hog_dir(im_dir, ft_dir) 

    im_dir = "{}/neg".format(imdir)
    ft_dir = "{}/neg".format(ftdir)
    _hog_dir(im_dir, ft_dir) 


def t1_hog():
    im_dir = "E:/0workspace/python/ml/hog_svm/image/person/"
    ft_dir = "E:/0workspace/python/ml/hog_svm/feat/person/8.3/64x128/"
    test(im_dir, ft_dir)

        
if __name__=='__main__':
    t1_hog()
    print('exit')
