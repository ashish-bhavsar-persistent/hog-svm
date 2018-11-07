from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import dconfig
   
def train_svm(pos_ft_dir, neg_ft_dir, mode_name='svm.model'): 
    fds = []   # features
    labels = []  # pos or neg labels

    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_ft_dir,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    for feat_path in glob.glob(os.path.join(neg_ft_dir,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
            
    print(np.array(fds).shape, len(labels))
    #clf = LinearSVC(C=dconfig.svm.C)
    clf = SVC(C=dconfig.svm.C, kernel=dconfig.svm.kernel)
    print("Training a {} SVM Classifier".format(dconfig.svm.kernel))
    clf.fit(fds, labels)

    joblib.dump(clf, mode_name)
    print("Done")


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


    
def debug_train_svm(pos_ft_dir, neg_ft_dir, mode_name='svm.model'):
    X = []   # features
    y = []  # pos or neg labels

    for feat_path in glob.glob(os.path.join(pos_ft_dir,"*.feat")):
        x = joblib.load(feat_path)
        X.append(x)
        y.append(1)

    for feat_path in glob.glob(os.path.join(neg_ft_dir,"*.feat")):
        x = joblib.load(feat_path)
        X.append(x)
        y.append(0)


    title = "Learning Curves (SVM, {} kernel)".format(dconfig.svm.kernel)
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    estimator = SVC(C=dconfig.svm.C, kernel=dconfig.svm.kernel, gamma=dconfig.svm.gamma)
    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
    
    plt.show()

def test(ftdir, debug=False):
    neg_ft_dir = "{}/neg/".format(ftdir)
    pos_ft_dir = "{}/pos/".format(ftdir)
    
    if debug == False:
        train_svm(pos_ft_dir, neg_ft_dir)
    else:
        debug_train_svm(pos_ft_dir, neg_ft_dir)
        
def t1():
    #ft_dir = "E:/0workspace/python/ml/hog_svm/feat/8.3/48x96/"
    #ft_dir = "E:/0workspace/python/ml/hog_svm/feat/8.3/64x128/"
    ft_dir = "E:/0workspace/python/ml/hog_svm/feat/person/8.3/64x128/"
    test(ft_dir)
    
if __name__=='__main__':
    t1()