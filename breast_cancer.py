import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('data/cancer.csv', index_col='id')
df.drop('Unnamed: 32', axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
cols = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean']
source = df[cols]
target = df['diagnosis']
classifier = RandomForestClassifier()
classifier.fit(source, target)
#ip = np.array([0.05,1001.0,1325.25,125.0,0.5]).reshape(1,5)
#y_pred = classifier.predict(ip)
#print(y_pred)
pickle.dump(classifier, open('cancer_model.pkl', 'wb'))
model = pickle.load(open('cancer_model.pkl', 'rb'))
