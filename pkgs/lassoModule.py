import numpy as np

class lassoRegression():
    # define the initial setting of lasso regression
    def __init__(self, learning_rate, iterations, l1_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penality = l1_penality


    def fit(self, X, Y):
        # X is the model input, which is an M by N matrix
        self.Mx, self.Nx = X.shape

        # weight initialization

        self.W = np.zeros(self.Nx) # W is the regression weights
        self.bias = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self


    def update_weights(self):
        Y_pred = self.predict(self.X)

        # calculate the gradient
        dW = np.zeros(self.Nx)

        for j in range(self.Nx):
            # since the gradient of L1 penalty has 2 possibilities 
            # (  d|Y|/dx  =  dY/dx  or  -dY/dx  )
            # the gradient would has 2 cases according to the sign symbol:  >0 or <0

            if self.W[j] > 0:
                # the first order differential
                dW[j] = (-(2*(self.X[:,j]).dot(self.Y - Y_pred)) 
                        + self.l1_penality) / self.Mx

            else:
                dW[j] = (-(2*(self.X[:,j]).dot(self.Y - Y_pred)) 
                        - self.l1_penality) / self.Mx

        dbias = -2*np.sum(self.Y - Y_pred) / self.Mx

        # update weights
        self.W = elf.W - self.learning_rate * dW
        self.bias = self.bias - self.learning_rate * dbias

        return self


    def predict(self, X):
        return X.dot(self.W) + self.bias