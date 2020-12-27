import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.over_sampling import SMOTE

df = pd.read_csv('data/indian_liver_patient.csv')
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df.drop('Albumin_and_Globulin_Ratio', axis=1, inplace=True)
source = df.drop(['Dataset', 'Alamine_Aminotransferase', 'Total_Protiens'], axis=1)
target = df['Dataset']
source, target = SMOTE().fit_sample(source, target)
classifier = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
classifier.fit(source, target)
print(source.columns)
pickle.dump(classifier, open('liver_model.pkl', 'wb'))
#liver_model = pickle.load(open('liver_model.pkl'), 'rb')
