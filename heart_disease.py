import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('data/heart.csv')
source = df.drop(['chol', 'fbs', 'target', 'trestbps'], axis=1)
target = df['target']
heart_classifier = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=80, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=600,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
heart_classifier.fit(source, target)
print(source.columns)
pickle.dump(heart_classifier, open('heart_model.pkl', 'wb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
