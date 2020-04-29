from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SVMSMOTE
from imblearn.under_sampling import AllKNN, TomekLinks, NearMiss, ClusterCentroids, OneSidedSelection, RandomUnderSampler, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, InstanceHardnessThreshold
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.datasets import make_classification

"""
Este archivo contiene, entre otras, las técnicas de
over y undersampling comentadas en el apartado 5.3 de Astralaria.pdf
"""


#OVERSAMPLING Y UNDERSAMPLING COMBINADOS
def smote_tomek(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res


def smote_enn(X, y):
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X, y)
    return X_res, y_res


#OVERSAMPLING
def smote(X, y):
    smote = SMOTE(random_state=0,n_jobs=12)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def adasyn(X, y):
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X, y)
    return X_res, y_res


def borderline_smote(X, y):
    sm = BorderlineSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def keans_smote(X, y):
    sm = KMeansSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def random_over_sampler(X, y):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res
     

def svm_smote(X, y):
    sm = SVMSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


#UNDERSAMPLING
def aiiknn(X, y):
    allknn = AllKNN()
    X_res, y_res = allknn.fit_resample(X,y)
    return X_res, y_res


def tomeklinks(X, y):
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)
    return X_res, y_res


def near_miss(X, y):
    nm = NearMiss()
    X_res, y_res = nm.fit_resample(X, y)
    return X_res, y_res


def cluster_centroids(X, y):
    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_resample(X, y)
    return X_res, y_res


def one_sided_selection(X, y):
    oss = OneSidedSelection(random_state=42)
    X_res, y_res = oss.fit_resample(X, y)
    return X_res, y_res


def random_under_sampler(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def condensed_nearest_neighbour(X, y):
    cnn = CondensedNearestNeighbour(random_state=42)
    X_res, y_res = cnn.fit_resample(X, y)
    return X_res, y_res


def edited_nearest_neighbour(X, y):
    enn = EditedNearestNeighbours()
    X_res, y_res = enn.fit_resample(X, y)
    return X_res, y_res


def repeated_edited_nearest_neighbours(X, y):
    renn = RepeatedEditedNearestNeighbours()
    X_res, y_res = renn.fit_resample(X, y)
    return X_res, y_res


def instance_hardness_thresold(X, y):
    iht = InstanceHardnessThreshold(random_state=42)
    X_res, y_res = iht.fit_resample(X, y)
    return X_res, y_res
    