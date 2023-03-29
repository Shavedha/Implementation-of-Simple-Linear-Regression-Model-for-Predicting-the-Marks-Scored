# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries and read the given .csv file into the program
2. Segregate the data into two variables (x-Hours and y-scores)
3. Split the data into Training and Test Data set
4. Import Linear Regression from sklearn display the predicted and the actual values.
5. Plot graph for both Training and Test dataset using matplot library.
6. Finally find the Mean Square Error,Mean absolute Error,Root Mean Square Error.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Y SHAVEDHA
RegisterNumber:  212221230095
```
```
#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

#READING THE .csv FILE
df=pd.read_csv('student_scores.csv')
print('df.head')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

# splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#PERFORMING LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#DISPLAYING THE PREDICTED VALUES
y_pred

#DISPLAYING THE ACTUAL VALUES
y_test

#GRAPH PLOT FOR TRAINING DATA
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#GRAPH PLOT FOR TEST DATA
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="orange")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#FINDING THE MSE,MAE AND RMSE
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RSME = ",rmse)

```


## Output:
### Reading the data set
<img width="547" alt="op1" src="https://user-images.githubusercontent.com/93427376/228585876-02577dff-b89b-4092-b0f3-502bbf72144d.png">
<img width="401" alt="op2" src="https://user-images.githubusercontent.com/93427376/228587329-84d2ea76-562e-46aa-b258-cf8033982419.png">

### Segregating Data into variables
<img width="404" alt="op3" src="https://user-images.githubusercontent.com/93427376/228587293-f4ff4d32-3353-41f1-925c-228741d7741f.png">
<img width="568" alt="op4" src="https://user-images.githubusercontent.com/93427376/228587383-70fda72c-c49b-4ed9-b292-a73b5e15b26c.png">

### Predicted Y values
<img width="561" alt="op5" src="https://user-images.githubusercontent.com/93427376/228587619-a05d3b55-45f5-432b-a396-b8a6a527f802.png">

### Actual Y values
<img width="510" alt="op6" src="https://user-images.githubusercontent.com/93427376/228587660-12f68500-d41b-49e2-a83f-5e39119cddd4.png">

### Graph for Training Data
<img width="548" alt="op7" src="https://user-images.githubusercontent.com/93427376/228587985-35d7aa6a-cf42-4940-83cb-328552d88cf0.png">

### Graph for Test Data
<img width="542" alt="op8" src="https://user-images.githubusercontent.com/93427376/228588027-ef4d2f49-cfd3-478c-b220-78b64644f45b.png">

### Finding the Errors
<img width="439" alt="op9" src="https://user-images.githubusercontent.com/93427376/228588210-bd302b03-0d87-4443-8a0f-d3db63ed9ed8.png">




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
