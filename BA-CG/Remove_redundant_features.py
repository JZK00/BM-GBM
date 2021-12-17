import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn import preprocessing

################# read_csv ################
data1 = pd.read_csv('./data/output/Remove redundant features.csv')#color images
# data = pd.read_csv('*.csv',index_col=0,encoding= 'unicode_escape')#SirRunRunShaw_feature(2)
# X = data.iloc[:,1:]
# y = data.iloc[:,0]
# print(X.shape)
# test = pd.read_csv('*.csv',index_col=0,encoding= 'unicode_escape')
# X_test = test.iloc[:,1:]
# y_test = test.iloc[:,0]
# print(X_test.shape)
#
# ################# StandardScaler ################
# scaler = preprocessing.StandardScaler().fit(X)
# X_data_transformed = scaler.transform(X)
# X_data_transformed=pd.DataFrame(X_data_transformed)
# X_data_transformed.columns=X.columns
# X_data=X_data_transformed
# X = X_data
# scaler = preprocessing.StandardScaler().fit(X_test)
# X_test_transformed = scaler.transform(X_test)
# X_test_transformed=pd.DataFrame(X_test_transformed)
# X_test_transformed.columns=X_test.columns
# X_test=X_test_transformed
################ StratifiedShuffleSplit ###########
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3
                               , random_state=66
                               )
for train_index, test_index in split.split(data1.iloc[:, 1:], data1.iloc[:, 0]):
    train_set = data1.iloc[train_index]
    test_set = data1.iloc[test_index]
print(len(train_set), len(test_set))
print(train_set['label'].value_counts() / len(train_set))

X = train_set.iloc[:,1:]
y = train_set.iloc[:,0]
X_test = test_set.iloc[:,1:]
y_test = test_set.iloc[:,0]

# data1 = pd.concat([y,X],axis=1)
# data2 = pd.concat([y_test,X_test],axis=1)
# data1.to_csv('*.csv')
# data2.to_csv('*.csv')

########## spearman #########
x_cols = [col for col in X.columns if X[col].dtype != 'object']
labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(abs(X[col].corr(y, 'spearman')))  # np.corrcoef(X[col].values,y.values)[0,1]
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})

# ############### Remove redundant #########################
features = corr_df['col_labels']
feature_matrix = X[features]
corr_matrix =feature_matrix.corr(method='spearman')#method='spearman'
remove_features = []
mask = (corr_matrix.iloc[:,:].values>0.9) & (corr_matrix.iloc[:,:].values<1)
for idx_element in range(len(corr_matrix.columns)):
    for idy_element in range(len(corr_matrix.columns)):
        if mask[idx_element,idy_element]:
#             print(idx_element,idy_element)
            if list(corr_df['corr_values'])[idx_element] > list(corr_df['corr_values'])[idy_element]:
                remove_features.append(list(features)[idy_element])
            else:
#                 print(list(features)[idx_element])
                remove_features.append(list(features)[idx_element])
remove_features = set(remove_features)
print(len(remove_features))
print(type(features))
print(type(remove_features))
remain_features = set(features) - remove_features
print(remain_features)
#
# #保存特征表
X = X.loc[:,remain_features]
data1 = pd.concat([y,X],axis=1)
X_test = X_test.loc[:,remain_features]
data2 = pd.concat([y_test,X_test],axis=1)
data1.to_csv('./data/output/remove_feature_t1_train.csv')
data2.to_csv('./data/output/remove_feature_t2_test.csv')
