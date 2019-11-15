# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
def collect_training_data(no_features,no_observations):
    data=[]
    for i in range(0,no_observations):
        data.append(pd.to_numeric(input().split(' ')))
    #data=array(data)
    data=np.asarray(data)
    #data=pd.to_numeric(data)
    data=pd.DataFrame(data)
    price_sqft=data.iloc[:,-1]
    data=data[price_sqft<=1000000]
    data=data[price_sqft>=0]
    training_data=data
    return training_data

def collect_test_data(no_features,no_test_cases):
    data=[]
    for i in range(0,no_test_cases):
        data.append(pd.to_numeric(input().split(' ')))
    #data=array(data)
    data=np.asarray(data)
    data=pd.DataFrame(data)
    #data=pd.to_numeric(data)
    test_data=data
    return test_data

def regression_analysis(training_data,test_data):
    Y_train=training_data.iloc[:,-1]
    Y_train=Y_train.values
    Y_train=np.asarray(Y_train)
    Y_train=Y_train.reshape(len(Y_train),)
    test_data=test_data.values
    columnNames = list(training_data.head(0)) 
    last_col_name=columnNames[-1]
    X_train=training_data.loc[:, training_data.columns != last_col_name]
    X_train=X_train.values
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    test_data=scaler.transform(test_data)
    #regr_1=DecisionTreeRegressor(criterion='mae',max_depth=10,random_state=0)
    #regr_2=AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=300,random_state=None)
    regr_3 = LinearRegression().fit(X_train, Y_train)
    #Y1=regr_1.fit(X_train,Y_train)
    #Y2=regr_2.fit(X_train,Y_train)
    #predicted_1=Y1.predict(test_data)
    #predicted_2=Y2.predict(test_data)
    predicted_3=regr_3.predict(test_data)
    print(*predicted_3,sep='\n')

first_line=input()
first_line=first_line.split(' ')
no_features=int(first_line[0])
no_observations=int(first_line[1])
if(no_features>=1 and no_features<=10):
    if(no_observations>=5 and no_observations<=100):
        training_data=collect_training_data(no_features,no_observations)
no_test_cases=int(input())
if(no_test_cases>=1 and no_test_cases<=100):
    test_data=collect_test_data(no_features,no_test_cases)

regression_analysis(training_data,test_data)

