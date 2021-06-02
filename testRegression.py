import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import Lasso
from pkgs.lassoModule import lassoRegression

# load the iris dataset
X = load_iris()['data']
Y = load_iris()['target']
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/3)

# load the from scratch model and sklearn model 
model = lassoRegression(learning_rate = 0.01, iterations = 1000, l1_penality = 0.5 )
slmodel = Lasso(max_iter=1000, alpha=0.5)

# model fit
model.fit( X_train, Y_train)
slmodel.fit( X_train, Y_train)

# prediction
Y_pred = model.predict(X_test)
slY_pred = slmodel.predict(X_test)

# compare the results
print('\n\n')
print( "from scratch predicted values :", np.round( Y_pred[:3], 2)) 
print( "sklearn predicted values      :", np.round( slY_pred[:3], 2)) 
print( "Real values                   :", Y_test[:3] )
