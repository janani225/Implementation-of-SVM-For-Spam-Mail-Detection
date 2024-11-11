# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.


## Program:
#### Program to implement the SVM For Spam Mail Detection..
#### Developed by:JANANI.V.S
#### RegisterNumber:212222230050
```
import pandas as pd
data= pd.read_csv("C:/Users/admin/Desktop/INTR MACH/spam.csv", encoding= 'Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train , y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test, y_pred)
accuracy
  

```

## Output:
#### data_head():
![image](https://github.com/user-attachments/assets/b73ce5e9-57c9-4f34-8219-025a668a6dfd)

#### data.isnull().sum():
![image](https://github.com/user-attachments/assets/146cbfe1-a52a-45c5-8925-e9b9d582b7fd)

 #### accuracy:
 ![image](https://github.com/user-attachments/assets/12ddb95f-fa69-4fb4-8393-365f1ab707e7)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

