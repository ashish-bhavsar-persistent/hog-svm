from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import dconfig
from my_common import plot_learning_curve

def load_data(pos_ft_dir, neg_ft_dir):
    print("data loading...")
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

    print("data loaded")
    return X, y


def train_svm(pos_ft_dir, neg_ft_dir, mode_name='svm.model', test=False):
    
    X, y = load_data(pos_ft_dir, neg_ft_dir)

    if test:
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=dconfig.spliter.test_size, 
                random_state=dconfig.spliter.random_state)
        print("len(X_train):{}, len(y_train):{}".format(len(X_train), len(y_train)))

    print("Training a {} SVM Classifier...".format(dconfig.svm.kernel))
    clf = SVC(C=dconfig.svm.C, kernel=dconfig.svm.kernel)
    clf.fit(X_train, y_train)

    joblib.dump(clf, mode_name)
    print("model saved as:{}".format(mode_name))

    if test:    
        score = clf.score(X_test, y_test)
        print("model score:{}".format(score))
    
    
def debug_train_svm(pos_ft_dir, neg_ft_dir, mode_name='svm.model'):
    X, y = load_data(pos_ft_dir, neg_ft_dir)
    
    title = "Learning Curves (SVM, {} kernel)".format(dconfig.svm.kernel)
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    estimator = SVC(C=dconfig.svm.C, kernel=dconfig.svm.kernel, gamma=dconfig.svm.gamma)
    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
    
    plt.show()


if __name__=='__main__':
    #ft_dir = "E:/0workspace/python/ml/hog_svm/feat/person/8.3/64x128/"
    #ft_dir = "E:/0workspace/python/ml/hog_svm/feat/8.3/64x128/"

    ft_dir = "E:/0workspace/python/ml/hog_svm/feat/8.3/48x96/"
    neg_ft_dir = "{}/neg/".format(ft_dir)
    pos_ft_dir = "{}/pos/".format(ft_dir)
    train_svm(neg_ft_dir, pos_ft_dir)