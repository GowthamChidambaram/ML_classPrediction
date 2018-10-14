import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
df=pd.read_csv("Classified Data",index_col=0)
print(df.head())


#standard scalling
scalar=StandardScaler()
scalar.fit(df.drop("TARGET CLASS",axis=1))
scaled_feautures=scalar.transform(df.drop("TARGET CLASS",axis=1))
print(scaled_feautures)
df_feautures=pd.DataFrame(data=scaled_feautures,columns=df.columns[:-1])
print(df_feautures.head())
x=df_feautures
y=df["TARGET CLASS"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#taking k=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print("result for nearest neighbour to be 1(k=1):")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#finding the best k
error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred=knn.predict(x_test)
    error_rate.append(np.mean(y_test!=pred))

fig=plt.Figure(figsize=(10,6))
plt.plot(range(1,40),error_rate)
plt.show()

#found the best k=36
knn = KNeighborsClassifier(n_neighbors=36)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print("result after finding the best k to be 36(k=36)")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

