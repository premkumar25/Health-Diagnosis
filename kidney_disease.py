import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import pickle

df = pd.read_csv('data/kidney_disease.csv', index_col='id')
df.replace(to_replace={'\t43':'43','\t6200':'6200','\t8400':'8400','\t?':np.nan,'ckd\t':1,'ckd':1,'notckd':0},inplace=True)
map_values = {'normal':0,'abnormal':1,'notpresent':0,'present':1,'yes':1,'no':0,'\tno':0,'\tyes':1,' yes':1,'good':1,'poor':0}
df.replace(to_replace=map_values,inplace=True)
imputer = KNNImputer(missing_values=np.nan)
df =pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
df = df.astype(float)
source = df[['sg','al','rbc','hemo','pcv','rc','htn','dm']]
target = df['classification']
classifier = RandomForestClassifier()
classifier.fit(source, target)
pickle.dump(classifier, open('kidney_model.pkl', 'wb'))
