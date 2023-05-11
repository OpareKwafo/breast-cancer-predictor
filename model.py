#import libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

#read data
df = pd.read_csv("breastCancer.csv")
df.drop('id', axis= 1, inplace=True)

#Explore data
df = df.replace('?', np.nan)
df = df.fillna(df.median())
df['bare_nucleoli'] = df['bare_nucleoli'].astype('int32')

#splitting data
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#The model
knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

#metrics
print(classification_report(predictions, y_test))

cm = confusion_matrix(predictions, y_test, labels=knn.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=knn.classes_)
disp.plot()


#aving our model
pickle.dump(knn, open('model.pkl', 'wb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))
print(loaded_model.predict([[5, 1, 1, 1, 2, 1, 3, 1, 1]]))
