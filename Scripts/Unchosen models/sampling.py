from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SVMSMOTE
from imblearn.under_sampling import AllKNN, TomekLinks, NearMiss, ClusterCentroids, OneSidedSelection, RandomUnderSampler, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, InstanceHardnessThreshold
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.datasets import make_classification
from visualization import hist_over_and_undersampling, pca_general

#OVERSAMPLING AND UNDERSAMPLING COMBINED
def smote_tomek(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def smote_enn(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


#OVERSAMPLING
def smote(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    smote = SMOTE(random_state=0,n_jobs=12)
    X_res, y_res = smote.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def adasyn(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def borderline_smote(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    sm = BorderlineSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def keans_smote(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    sm = KMeansSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def random_over_sampler(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res
     

def svm_smote(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    sm = SVMSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


#UNDERSAMPLING
def aiiknn(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    allknn = AllKNN()
    X_res, y_res = allknn.fit_resample(X,y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def tomeklinks(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def near_miss(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    nm = NearMiss()
    X_res, y_res = nm.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def cluster_centroids(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def one_sided_selection(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    oss = OneSidedSelection(random_state=42)
    X_res, y_res = oss.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def random_under_sampler(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def condensed_nearest_neighbour(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    cnn = CondensedNearestNeighbour(random_state=42)
    X_res, y_res = cnn.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def edited_nearest_neighbour(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    enn = EditedNearestNeighbours()
    X_res, y_res = enn.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def repeated_edited_nearest_neighbours(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    renn = RepeatedEditedNearestNeighbours()
    X_res, y_res = renn.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res


def instance_hardness_thresold(X, y, visualize = False, pca2d=True, pca3d=True, tsne=True, pie_evr=True):
    iht = InstanceHardnessThreshold(random_state=42)
    X_res, y_res = iht.fit_resample(X, y)
    if visualize == True:
        hist_over_and_undersampling(y_res)
        pca_general(X_res, y_res, d2=pca2d, d3=pca3d, pie_evr=pie_evr)
    return X_res, y_res
    