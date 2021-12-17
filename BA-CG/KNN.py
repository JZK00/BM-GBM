import pandas as pd
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest
#import pymrmr
import numpy as np
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import statistics
from sklearn.metrics import plot_roc_curve
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
from sklearn import neighbors
NUM_RUNS = 30 # The number of trials
NUM_JOBS=8  # The number of threads for parallelization. Each CPU core can handle 2 threads
seed=12345 # seed used by the random number generator. Used for re-producability

data = pd.read_csv('*.csv',index_col=0)#SirRunRunShaw_feature(2)
X_data = data.iloc[:,1:]
y_data = data.iloc[:,0]
print(X_data.shape)
test = pd.read_csv('*.csv',index_col=0)#SirRunRunShaw_feature(2)
X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]
print(X_test.shape)


############### 调参 ##########
classifier_default = neighbors.KNeighborsClassifier()
parameters_grid = {"n_neighbors": np.arange(2, 20, 1),
     "weights": ["uniform", "distance"]
  }

# Arrays to store scores
grid_search_best_scores = np.zeros(NUM_RUNS)  # numpy arrays
final_evaluation_scores = np.zeros(NUM_RUNS)  # numpy arrays

for i in range(NUM_RUNS):
    folds_for_grid_search = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    folds_for_evaluation = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Parameter tuning with grid search and cross validation
    tuned_model = GridSearchCV(estimator=classifier_default, param_grid=parameters_grid, cv=folds_for_grid_search,
                               scoring='roc_auc',
                               n_jobs=-1)
    # n_jobs=Number of jobs to run in parallel
    tuned_model.fit(X_data, y_data)
    grid_search_best_scores[i] = tuned_model.best_score_

    # print (tuned_model.best_score_)
    print()
    print("Best Selected Parameters:")
    print(tuned_model.best_params_)

    y_pred = cross_val_predict(tuned_model.best_estimator_, X_data, y_data, cv=folds_for_evaluation)

    print()
    print("Classification Results")
    print(classification_report(y_data, y_pred))

############ ROC ################
from sklearn.metrics import plot_roc_curve

cv = StratifiedKFold(n_splits=5, shuffle=True
                     , random_state=1
                     )
classifier = neighbors.KNeighborsClassifier(**tuned_model.best_params_)  # **tuned_model.best_params_C=2, max_iter=1000, penalty='l2',solver='sag'

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X_data, y_data)):
    xtr, xvl = X_data.iloc[train], X_data.iloc[test]
    ytr, yvl = y_data[train], y_data[test]

    classifier.fit(xtr, ytr)
    viz = plot_roc_curve(classifier, xvl, yvl
                         , name='ROC fold {}'.format(i)
                         , alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
# fig=plt.gcf()
plt.show()
# fig.savefig('./r.png')
# fig.savefig('./r.jpg')
result = classifier.predict(X_test)
score = classifier.score(X_test,y_test)
recall = recall_score(y_test, result)
precision = precision_score(y_test, result)
auc = roc_auc_score(y_test,classifier.predict_proba(X_test)[:, 1])
print(classification_report(y_test, result))
plot_roc_curve(classifier,X_test,y_test)  # testing accuracy 0.772727, recall is 0.750000', auc is 0.802083
plt.show()
print("testing accuracy %f, recall is %f', auc is %f,precision is %f" % (score,recall,auc,precision))#rbf 's
tn, fp, fn, tp = metrics.confusion_matrix(y_test, result).ravel()
print(tn, fp, fn, tp)
