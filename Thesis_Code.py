# -*- coding: utf-8 -*-
"""
# New Section
"""

import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

DrDoS_NTP_data_data_5_per = pd.read_csv('/content/drive/MyDrive/DrDoS_NTP.csv')

data = pd.concat([DrDoS_NTP_data_data_5_per],ignore_index = True)

data.shape

data[' Label'].value_counts()

data.columns

data_real = data.replace(np.inf, np.nan)

data_real.isnull().sum().sum()

data_df = data_real.dropna(axis=0)

data_df.isnull().sum().sum()

data_df

data_X = data_df.drop([' Label', 'SimillarHTTP'], axis = 1)

data_X.columns

data_X.shape

data_y = data_df[' Label']

data_y.shape

data_df.isnull().sum().sum()

data_y.unique()

data_X

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data_y_trans = le.fit_transform(data_y)

data_y_trans

le_fid = LabelEncoder()

le_fid.fit(data_X['Flow ID'])
data_X['Flow ID'] = le_fid.fit_transform(data_X['Flow ID'])

le_SIP = LabelEncoder()

le_SIP.fit(data_X[' Source IP'])
data_X[' Source IP'] = le_SIP.fit_transform(data_X[' Source IP'])

le_DIP = LabelEncoder()

le_DIP.fit(data_X[' Destination IP'])
data_X[' Destination IP'] = le_DIP.fit_transform(data_X[' Destination IP'])

le_timestamp = LabelEncoder()
le_timestamp.fit(data_X[' Timestamp'])
data_X[' Timestamp'] = le_timestamp.fit_transform(data_X[' Timestamp'])

data_X

data_X.dtypes

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier

#selecting 20 best features
# select_best= SelectKBest(chi2, k=20)
# X_feat_20 = select_best.fit_transform(data_X, data_y_trans)
# X_feat_20.shape

model = ExtraTreesClassifier(random_state=42)
model.fit(data_X, data_y_trans)

model.feature_importances_

feature_importance_std = pd.Series(model.feature_importances_, index=data_X.columns)
feature_importance_std.nlargest(20).plot(kind='bar', title='Standardised Dataset Feature Selection using ExtraTreesClassifier')

data_X.shape

data_new_20features_X = data_X[[' Timestamp', ' Source Port', ' Min Packet Length', ' Fwd Packet Length Min', 'Flow ID', ' Packet Length Mean', ' Fwd Packet Length Max', ' Average Packet Size', ' ACK Flag Count', ' Avg Fwd Segment Size', ' Fwd Packet Length Mean', 'Flow Bytes/s', ' Max Packet Length', ' Protocol', 'Fwd Packets/s', ' Flow Packets/s', 'Total Length of Fwd Packets', ' Subflow Fwd Bytes', ' Destination Port', ' act_data_pkt_fwd']]

data_new_20features_X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y_trans, test_size = 0.30, random_state = 42)

X_train.shape

X_test.shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)

from sklearn.model_selection import train_test_split
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(data_new_20features_X, data_y_trans, test_size = 0.30, random_state = 42)

from sklearn.preprocessing import StandardScaler
ss_20 = StandardScaler()
X_train_std_20 = ss_20.fit_transform(X_train_20)
X_test_std_20 = ss_20.fit_transform(X_test_20)

X_train_std_20.shape

y_train_20.shape

X_test_std_20.shape

y_test_20.shape

from sklearn.ensemble import GradientBoostingClassifier

gradinet_boost = GradientBoostingClassifier()
gradinet_boost.fit(X_train_std_20, y_train_20)

y_pred_xgboost = gradinet_boost.predict(X_test_std_20)

y_pred_xgboost

y_test_20

from sklearn.metrics import accuracy_score

print("Accuracy Score for the XGBoost Classifier is: {0:.3f}%".format(accuracy_score(y_test_20, y_pred_xgboost)* 100))

from sklearn.metrics import classification_report

print("Classification Report for XGBOOST: \n", classification_report(le.inverse_transform(y_test_20), le.inverse_transform(y_pred_xgboost)))

from sklearn.metrics import confusion_matrix

xgboost_conf_mat = confusion_matrix(y_test_20, y_pred_xgboost)
print("LSTM Confusion: \n", xgboost_conf_mat)

from sklearn.metrics import roc_curve

from sklearn import metrics

from matplotlib import pyplot as plt

# RoC Curve
title = 'Receiver operating characteristic of XGBOOST'
fpr, tpr, _ = metrics.roc_curve(y_test_20, y_pred_xgboost)
auc = metrics.roc_auc_score(y_test_20, y_pred_xgboost)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

len(y_pred_xgboost)

!pip install scikitplot

pip install scikit-plot

import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_test_20, y_pred_xgboost)
plt.show()

