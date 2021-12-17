import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV


################ Importing data and pre-processing ##########
#Importing data
data_x = pd.read_csv('*.csv',index_col=0)
X_x = data_x.iloc[:,1:]
y_x = data_x.iloc[:,0]
# X_d = data_d.iloc[:,1:]
# y_d = data_d.iloc[:,0]

#########Lasso festure screening#######
alphas = np.logspace(-4,1,100)
model_lassoCV = LassoCV(alphas = alphas, max_iter = 100000).fit(X_x,y_x)
coef = pd.Series(model_lassoCV.coef_, index = X_x.columns)
print(model_lassoCV.alpha_)
print('%s %d '%('Lasso picked',sum(coef != 0)))
index = coef[coef != 0].index
X_x_raw = X_x
X_x = X_x[index]

X_x.to_csv('./Lasso.csv')






