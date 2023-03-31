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
<img width="255" alt="image" src="https://user-images.githubusercontent.com/93427376/229055176-a8118fe4-b541-4970-9a0e-b860d3f43d44.png">
<img width="234" alt="image" src="https://user-images.githubusercontent.com/93427376/229055240-328abc6d-4a6d-46c7-91f0-8c2f3392fdc5.png">


### Segregating Data into variables
<img width="340" alt="image" src="https://user-images.githubusercontent.com/93427376/229055740-4b5aae1f-325e-4ce0-906e-1aa178d6ec69.png">
<img width="537" alt="image" src="https://user-images.githubusercontent.com/93427376/229055956-7ac711cb-f9ec-4070-b8b1-cb7442d93d12.png">

### Predicted Y values
<img width="505" alt="image" src="https://user-images.githubusercontent.com/93427376/229056059-5203afb5-67cc-4d99-a6bd-e5facf73d681.png">

### Actual Y values
<img width="483" alt="image" src="https://user-images.githubusercontent.com/93427376/229056131-bda4990e-3604-445b-b891-395e370985d5.png">

### Graph for Training Data
<img width="482" alt="image" src="https://user-images.githubusercontent.com/93427376/229056303-d6283810-d45b-4994-a577-564e961b4cf4.png">

### Graph for Test Data
<img width="462" alt="image" src="https://user-images.githubusercontent.com/93427376/229056379-c370fd43-705f-4c11-9912-e70db105612d.png">

### Finding the Errors
<img width="256" alt="image" src="https://user-images.githubusercontent.com/93427376/229056491-9a730504-f1f4-4e32-bff5-5a2fa26c85df.png">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
