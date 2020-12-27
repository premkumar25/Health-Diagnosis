import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('data/diabetes.csv', index_col='p_id')
cols = ['serum_insulin', 'glucose_concentration', 'blood_pressure', 'skin_fold_thickness', 'bmi']
for i in cols:
    df[i] = df[i].mask(df[i] == 0, df[i].mean())
source = df.drop('diabetes', axis=1)
target = df['diabetes']
classifier = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=30, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, n_estimators=1000,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
classifier.fit(source, target)
pickle.dump(classifier, open('diabetes_model.pkl', 'wb'))