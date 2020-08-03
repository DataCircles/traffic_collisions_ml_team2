import os
import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Encode
from sklearn.preprocessing import LabelBinarizer

# ML
from sklearn.tree import DecisionTreeClassifier

# Ensemble method
from sklearn.ensemble import RandomForestClassifier

# Split
from sklearn.model_selection import train_test_split

# Metric to evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Standard Scaler
from sklearn.preprocessing import StandardScaler

# What causes a more severe accident?
# Data dictionary https://www.seattle.gov/Documents/Departments/SDOT/GIS/Collisions_OD.pdf
X_org = pd.read_csv("/Users/ou/Projects/traffic_collisions_ml_team2/data/Collisions.csv")

# Unique id identify each accident
X_org['INCKEY'].duplicated().value_counts()
X_org['COLDETKEY'].duplicated().value_counts()

# Drop 'LOCATION'
# Drop 'REPORTNO', 'STATUS', EXCEPTRSNCODE, EXCEPTRSNDESC ST_COLCODE
# Convert  INCDATE, INCDTTM
X = X_org[['SEVERITYDESC', 'ADDRTYPE', 'EXCEPTRSNCODE', 'EXCEPTRSNDESC', 'COLLISIONTYPE', 'PERSONCOUNT',
       'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INJURIES', 'SERIOUSINJURIES',
       'FATALITIES', 'INCDATE', 'INCDTTM', 'JUNCTIONTYPE', 'SDOT_COLCODE',
       'SDOT_COLDESC', 'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND',
       'LIGHTCOND', 'PEDROWNOTGRNT', 'SDOTCOLNUM', 'SPEEDING', 'ST_COLCODE',
       'ST_COLDESC', 'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR']]

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
num_col = ["PERSONCOUNT", "PEDCOUNT", "PEDCYLCOUNT", "VEHCOUNT", "INJURIES",
    "SERIOUSINJURIES", "FATALITIES"]
num_mask = X.columns.isin(num_col)
cat_col = X.columns[~num_mask].tolist()

# Fill missing values with 0
X[num_col] = X[num_col].apply(lambda x: x.astype(int).fillna(0)).copy()
X[cat_col] = X[cat_col].apply(lambda x: x.fillna('MISSING')).copy()
# Create LabelEncoder object: le
le = LabelEncoder()
# Apply LabelEncoder to categorical columns
X[cat_col] = X[cat_col].apply(lambda x: le.fit_transform(x.astype(str))).copy()
# Print the head of the LabelEncoded categorical columns
print(X[cat_col].head())


# Feature Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)

y = X['SEVERITYDESC'].to_frame().copy()
X = X.drop(['SEVERITYDESC'], axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

# Other encoding
# encoder3 = LabelBinarizer()
# encoder3.fit_transform(X['ADDRTYPE'])

# Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# tree_clf = DecisionTreeClassifier()
# tree_clf.fit(y_train, X_train)

# Naive Bayes

# Random forest classifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train.values.ravel())
y_pred = forest_clf.predict(X_test)

### Classification
## ask for precision, recall, accuracy, F1 Score
## Precision: precision = TP/ (TP + FP)
from sklearn.metrics import precision_score
precision_score = precision_score(y_test, y_pred, average=None)
print(precision_score)

## recall/ sensitivity/ true positive rate = TP/ (TP + FN)
from sklearn.metrics import recall_score
recall_score = recall_score(y_test, y_pred, average=None)
print(recall_score)

## accuracy
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_pred)
print(accuracy_score)

## F1 Score = F = 2/ (1/precision + 1/recall)
from sklearn.metrics import f1_score
f1_score = f1_score(y_test, y_pred, average=None)
print(f1_score)

pdb.set_trace()
