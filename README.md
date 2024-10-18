# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data**: Read and preprocess the dataset.
2. **Encode Variables**: Convert categorical features to numerical codes.
3. **Initialize Weights**: Randomly set initial parameters.
4. **Define Sigmoid Function**: Compute sigmoid activation.
5. **Define Loss Function**: Calculate cross-entropy loss.
6. **Implement Gradient Descent**: Update weights iteratively.
7. **Make Predictions**: Use weights to generate predictions.
8. **Calculate Accuracy**: Evaluate the accuracy of predictions.
9. **Test New Inputs**: Predict outcomes for new data points.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ASHWIN KUMAR A
RegisterNumber: 212223040021 
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop(['sl_no'],axis=1)
dataset=dataset.drop(['salary'],axis=1)
dataset
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta =gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
<br><b><br><b><br><b>
<br><b><br><b><br><b>
<br><b><br><b><br><b>
<br><b><br><b><br><b>
<br><b><br><b><br><b>
## Output:
![image](https://github.com/user-attachments/assets/4fba22b4-5dfd-4cbc-8b22-03b54b3fcf3e)
![image](https://github.com/user-attachments/assets/8f0acdd1-8a5f-41ee-816f-98718d3f89cc)
![image](https://github.com/user-attachments/assets/10073ff1-6b6d-4037-a7c3-dff6a8213dc8)
![image](https://github.com/user-attachments/assets/11836b3c-237e-4b49-a386-5d028ab69641)
![image](https://github.com/user-attachments/assets/ad604f90-6db9-4bc8-a0e1-bae5076b2458)
![image](https://github.com/user-attachments/assets/707a6d5d-b77e-49d0-b353-792147673ffc)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

