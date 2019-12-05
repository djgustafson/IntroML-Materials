#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:38:41 2018

@author: danielgustafson
"""

#%% Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing.imputation import Imputer
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing

from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

#%% Mass mobilization data
mm = pd.read_csv("/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/full_mm.csv")

#%% Separate into X and y
ids = mm.iloc[:,0:3]

X = mm.iloc[:,4:]

y = mm.protests.values

#%% Imputing the feature data 
imp = Imputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X_impute = imp.transform(X)

#%% Scale data
# Get column names first
names = list(X)
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
X_impute_scaled = scaler.fit_transform(X_impute)
X_impute_scaled = pd.DataFrame(X_impute_scaled, columns=names)

#%% Split the data
X_train, X_test, y_train, y_test= train_test_split(X_impute_scaled, y, 
                                                   test_size=0.2,
                                                   random_state=1523)

#%% LM
lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred_lm = lm.predict(X_test)

lm_mse = mse(y_test, y_pred_lm)

#%% RF
rf = RandomForestRegressor(n_estimators=1000, max_leaf_nodes=20, 
                            random_state=1523, n_jobs=-1)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_mse = mse(y_test, y_pred_rf)


#%% GBM
gbm = GradientBoostingRegressor(max_depth=1, n_estimators=1000, random_state=1523)

gbm.fit(X_train, y_train)

y_pred_gbm = gbm.predict(X_test)

gbm_mse = mse(y_test, y_pred_gbm)


#%%
params_cv = {'n_estimators': [300, 500, 1000],
             'max_depth': [1,2,3,4],
             'min_samples_leaf': [.01, .02,.05,.1]
             }

grid_cv = GridSearchCV(estimator=gbm,
                       param_grid=params_cv,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=2,
                       n_jobs=-1)

grid_cv.fit(X_train, y_train)


best_hyperparams = grid_cv.best_params_

print('Best hyerparameters:\n', best_hyperparams)
#%%

best_model = GradientBoostingRegressor(max_depth=4, min_samples_leaf=.02, n_estimators=300, random_state=1523)

best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

best_mse = mse(y_test, y_pred_best)

#%% MSE Plot
height = [rf_mse, gbm_mse, best_mse]
bars = ('RF', 'GBM', 'GBM-CV')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)

plt.title('Protest Model MSE')

plt.savefig('/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/protest_mse.png', dpi = 500)
 
# Show graphic
plt.show()

#%% Feature importance plot
var_imp=pd.DataFrame([np.array(list(X)), best_model.feature_importances_]).T

var_imp.columns = ['feature', 'importance']

var_imp = var_imp.sort_values('importance', ascending = False)

var_imp[:10]

#%%
height = var_imp['importance'][:10].tolist()
bars = ('Urb. Pop.', 'Lab. Force Ed.', 'Agriculture Growth', '% Male 25-29', 'Idustry Value', 'SA Merch. Imp.', 'Cap. Form. Gr.', 'Aqua. Prod.', 'Nat. Debt Int. Paym.', 'Serv. Gr.')

y_pos=np.arange(len(bars))

plt.bar(y_pos, height)

plt.xticks(y_pos, bars, rotation = 70)

plt.title('Protest Feature Importance')

plt.tight_layout()

plt.savefig('/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/protest_importance.png', dpi = 500)

plt.show()

#%%
# GDP
print(X.columns.get_loc('NY.GDP.PCAP.CD'))

# Consumer Price
print(X.columns.get_loc('FP.CPI.TOTL'))

# Unemployment
print(X.columns.get_loc('SL.UEM.TOTL.ZS'))

# GDP growth
print(X.columns.get_loc('NY.GDP.MKTP.KD.ZG'))

X.rename(columns={'SP.URB.TOTL':'Urban Population',
                          'SL.TLF.BASC.ZS':'Labor Force Education',
                          'DT.INT.DECT.GN.ZS':'Nat. Debt. Int. Paym.',
                          'NY.GDP.PCAP.CD':'GDP Per Capita',
                          'NY.GDP.MKTP.KD.ZG':'GDP Growth',
                          'SL.UEM.TOTL.ZS':'Unemployment'}, 
                 inplace=True)

#%% Partial Dependence!

names = list(X)
features = [1496, 1346, 233,850, 845, 1386]
fig, axs = plot_partial_dependence(best_model, X_train, features,
                                       feature_names=names,
                                       n_jobs=-1, grid_resolution=100)

fig.suptitle('Partial dependence of protest occurrence \n on important and popular features')
plt.subplots_adjust(top=.8, hspace =.5, wspace = .5)

plt.savefig('/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/protest_part_dep.png', dpi = 500)















#%% GTD data
gtd = pd.read_csv("/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/full_gtd.csv")

#%% Separate into X and y
ids = gtd.iloc[:,0:3]

X = gtd.iloc[:,4:]

y = gtd.attacks.values

#%% Imputing the feature data 
imp = Imputer(missing_values=np.nan, strategy='median')
imp.fit(X)
X_impute = imp.transform(X)

#%% Scale data
# Get column names first
names = list(X)
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
X_impute_scaled = scaler.fit_transform(X_impute)
X_impute_scaled = pd.DataFrame(X_impute_scaled, columns=names)

#%% Split the data
X_train, X_test, y_train, y_test= train_test_split(X_impute_scaled, y, 
                                                   test_size=0.2,
                                                   random_state=1523)

#%% LM
lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred_lm = lm.predict(X_test)

mse(y_test, y_pred_lm)

#%% RF
rf = RandomForestRegressor(n_estimators=1000, max_leaf_nodes=20, 
                            random_state=1523, n_jobs=-1)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_mse = mse(y_test, y_pred_rf)

#%% GBM
gbm = GradientBoostingRegressor(max_depth=1, n_estimators=1000, random_state=1523)

gbm.fit(X_train, y_train)

y_pred_gbm = gbm.predict(X_test)

gbm_mse = mse(y_test, y_pred_gbm)

#%%
params_cv = {'n_estimators': [300, 500, 1000],
             'max_depth': [1,2,3,4],
             'min_samples_leaf': [.01, .02,.05,.1]
             }

grid_cv = GridSearchCV(estimator=gbm,
                       param_grid=params_cv,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=2,
                       n_jobs=-1)

grid_cv.fit(X_train, y_train)


best_hyperparams = grid_cv.best_params_

print('Best hyerparameters:\n', best_hyperparams)
#%%

best_model = GradientBoostingRegressor(max_depth=3, min_samples_leaf=.01, n_estimators=500, random_state=1523)

best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

best_mse = mse(y_test, y_pred_best)

#%% MSE Plot
height = [rf_mse, gbm_mse, best_mse]
bars = ('RF', 'GBM', 'GBM-CV')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)

plt.title('Terrorism Model MSE')

plt.savefig('/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/terrorism_mse.png', dpi = 500)
 
# Show graphic
plt.show()

#%% Feature importance plot
var_imp=pd.DataFrame([np.array(list(X)), best_model.feature_importances_]).T

var_imp.columns = ['feature', 'importance']

var_imp = var_imp.sort_values('importance', ascending = False)

var_imp[:10]

#%%
height = var_imp['importance'][:10].tolist()
bars = ('Battle Deaths', 'Exp. Volume', 'Bil. Aid', 'Displ. Pers.', 'Ch. Inter. Arrears', 'Merch. Trade', 'Nat. Debt Int. Paym.', 'Fish. Prod.', 'SA Merch. Exp.', 'Vuln. Empl.')

y_pos=np.arange(len(bars))

plt.bar(y_pos, height)

plt.xticks(y_pos, bars, rotation = 70)

plt.title('Terrorism Feature Importance')

plt.tight_layout()

plt.savefig('/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/terrorism_importance.png', dpi = 500)

plt.show()

#%%

X.rename(columns={'VC.BTL.DETH':'Battle Deaths',
                          'TX.QTY.MRCH.XD.WD':'Export Volume',
                          'DT.INT.DECT.EX.ZS':'Nat. Debt. Int. Paym.',
                          'NY.GDP.PCAP.CD':'GDP Per Capita',
                          'NY.GDP.MKTP.KD.ZG':'GDP Growth',
                          'SL.UEM.TOTL.ZS':'Unemployment'}, 
                 inplace=True)

X.rename(columns={'DC.DAC.USAL.CD':'Bilateral Aid'}, 
                 inplace=True)

#%% Partial Dependence!

names = list(X)
features = [1563, 114, 232, 850, 845, 1386]
fig, axs = plot_partial_dependence(best_model, X_train, features,
                                       feature_names=names,
                                       n_jobs=-1, grid_resolution=100)


fig.suptitle('Partial dependence of terrorist attack \n occurrence on important and popular features')

plt.subplots_adjust(top=.8, hspace =.5, wspace = .5)

plt.savefig('/Users/danielgustafson/Documents/Grad/Fall 2018/Machine Learning/Final Project/terrorism_part_dep.png', dpi = 500)
