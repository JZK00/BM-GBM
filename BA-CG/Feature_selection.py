import pandas as pd
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest
# import pymrmr
import numpy as np
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import statistics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, \
    average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import auc
NUM_RUNS = 30 # The number of trials
NUM_JOBS=8  # The number of threads for parallelization. Each CPU core can handle 2 threads
seed=12345 # seed used by the random number generator. Used for re-producability

data = pd.read_csv('./data/output/remove_feature_t1_train.csv')#SirRunRunShaw_feature(2)
X = data.iloc[:,2:]
y_data = data.iloc[:,1]
print(X.shape)
test = pd.read_csv('./data/output/remove_feature_t1_test.csv')
X_test = test.iloc[:,2:]
y_test = test.iloc[:,1]
print(X_test.shape)

scaler = preprocessing.StandardScaler().fit(X)
X_data_transformed = scaler.transform(X)
X_data_transformed=pd.DataFrame(X_data_transformed)
X_data_transformed.columns=X.columns
X_data=X_data_transformed
scaler = preprocessing.StandardScaler().fit(X_test)
X_test_transformed = scaler.transform(X_test)
X_test_transformed=pd.DataFrame(X_test_transformed)
X_test_transformed.columns=X_test.columns
X_test=X_test_transformed
#
# ################ MIC ##########
result = MIC(X_data,y_data,random_state=100)
k = result.shape[0] - sum(result <= 0)
Select = SelectKBest(MIC,k=k)
Select.fit(X_data, y_data)
X_new =Select.transform(X_data)
# X_new = SelectKBest(chi2,k=131).fit_transform(X_data, y_data)
# X_data=X_new
print(X_new.shape)

X=X_data.T
X_data=X[Select.get_support()].T
print(X_data.shape)
X1=X_test.T
X_test=X1[Select.get_support()].T
print(X_test.shape)

# ################ RFE ##########
import warnings
warnings.filterwarnings('ignore')
model = SVC(kernel='linear', probability=True,random_state=seed)
rfe = RFE(model, 10)
fit = rfe.fit(X_data, y_data)
X_data_rfe = fit.transform(X_data)
X_test_rfe = fit.transform(X_test)
print(X_data_rfe.shape)
print(X_test_rfe.shape)
print(rfe.ranking_)

# model = LogisticRegression(random_state=seed)
# rfe = RFE(model, 10)
# fit = rfe.fit(X_data, y_data)
# X_data_rfe = fit.transform(X_data)
# X_test_rfe = fit.transform(X_test)
# print(X_data_rfe.shape)
# print(X_test_rfe.shape)


X2=X_data.T
X_data=X2[rfe.get_support()].T
print(X_data.shape)
X3=X_test.T
X_test=X3[rfe.get_support()].T
print(X_test.shape)
data1 = pd.concat([y_data,X_data],axis=1)
data2 = pd.concat([y_test,X_test],axis=1)
data1.to_csv('./data/output/remove_selection_feature_t1_train.csv')
data2.to_csv('./data/output/remove_selection_feature_t1_test.csv')
