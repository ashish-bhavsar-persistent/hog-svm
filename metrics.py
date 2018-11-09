from skimage import io,color
import os 
from sklearn.externals import joblib
import numpy as np
import dconfig
from my_common import my_hog

def predict_im(clf, file_path): 
    
    im = io.imread(file_path)
    #TODO: check image's widht and hegiht here
    
    if dconfig.hog.multichannel == False:
        im = color.rgb2gray(im)
    
    fd = my_hog(im)
    fd = fd.reshape(1, -1)
    pred = clf.predict(fd)
    decision = clf.decision_function(fd) 
    return pred, decision

def process_im_folder(clf, im_dir):
    print('processing {}'.format(im_dir))

    result=[]    
    file_list = os.listdir(im_dir)
    counter = 0
    for f in file_list:
        (filename, extension) = os.path.splitext(f)
        if extension == '.png' or \
            extension == '.jpg' or \
            extension == '.pgm':
            file_path = os.path.join(im_dir, f)
            if os.path.isfile(file_path):
                pred, decision = predict_im(clf, file_path)
                result.append((filename, pred, decision))
                counter += 1
                if counter % 100 == 0:
                    pos_num = [n for (n, p, d) in result if p == 1]
                    valid_pos_num = [n for (n, p, d) in result 
                                     if p == 1 if d > dconfig.detector.decision]
                    neg_num = [n for (n, p, d) in result if p == 0]
                    print('finished {:.2f}, pos:{}, neg:{}, valid_pos:{}'.format(
                            counter/len(file_list), len(pos_num), len(neg_num), len(valid_pos_num)))
    
    return result

def metrics(pos_dir, neg_dir):
    clf = joblib.load(dconfig.svm.model) 
    pos_result = process_im_folder(clf, pos_dir)
    neg_result = process_im_folder(clf, neg_dir)

    #Y = np.hstack((np.ones(len(pos_result)), np.zeros(len(neg_result))))
    #oY = 
       
    TP = [f for (f, p, d) in pos_result if p == 1]
    FP = [f for (f, p, d) in pos_result if p == 0]
    TN = [f for (f, p, d) in neg_result if p == 0]
    FN = [f for (f, p, d) in neg_result if p == 1]

    accuracy = (len(TP) + len(TN)) / (len(TP) +len(TN) + len(FP) +len(FN))
    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))
    print('accuracy:{}, precision:{}, recall:{}'.format(accuracy, precision, recall))


if __name__ == '__main__':
    pos_dir = 'E:/0workspace/python/ml/hog_svm/image/48x96/pos'
    neg_dir = 'E:/0workspace/python/ml/hog_svm/image/48x96/neg'

    metrics(pos_dir, neg_dir)

