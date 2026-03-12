import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error
current_path = os.path.dirname(__file__)
file_path = os.path.join(current_path,"..",'data','sleep_mobile_stress_dataset_15000.csv')

file_path = os.path.abspath(file_path)

data = pd.read_csv(file_path)

#print('Categories',data.columns.tolist())

X = data.drop(columns=['stress_level','user_id'])
y = data['stress_level']


categorical_features = X.select_dtypes(include=['category','object']).columns.tolist()
numeric_features = X.select_dtypes(include=['int32','float64']).columns.tolist()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21,shuffle=True)

preprocessing = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric_features),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ],
    remainder='drop'
)

pipeline  = Pipeline(
    [
        ('preprocessing',preprocessing),
        ('regression',LinearRegression())
    ]
)

pipeline.fit(X_train,y_train)
y_predict = pipeline.predict(X_test)

mse = mean_squared_error(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
print('Prediction',y_predict[:10])
print('R2 Test',pipeline.score(X_test,y_test))
print('R2 Train',pipeline.score(X_train,y_train))
print('mean squared error',mse)
print('mean absolute error',mae)

