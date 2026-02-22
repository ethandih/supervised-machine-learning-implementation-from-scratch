import numpy as np

class LogisticRegression():
    
    def __init__(self, L=0.01,epochs=50): 
        self.L = L
        self.epochs = epochs
        self.losses, self.train_accuracies = [], []
        self.weights, self.bias = None, 4 #private: vector<float> weights;
        pass

    def sigmoid_function(self, z): #Maps the values of z into domain [0,1]
        return 1 / (1 + np.exp(-z))
        # return z
    
    def _compute_loss(self, y, y_pred):
        epsilon = 0# 1e-15 #small value to prevent zero division errors
        return -np.mean(y*np.log(y_pred+epsilon) + (1-y)*np.log(1-y_pred+epsilon))

    def compute_gradients(self, x, y, y_pred, nSamples):
        grads_w = (1/nSamples) * np.dot(x.T, (y_pred - y))
        grad_b = (1/nSamples) * np.sum(y_pred - y)
        return grads_w, grad_b

    def update_parameters(self, grads_w, grad_b):
        self.weights -= self.L*grads_w
        self.bias -= self.L*grad_b
    
    def lin_model(self, x):
        return np.dot(x, self.weights) + self.bias

    def accuracy(self, true_values, predictions):
        return np.mean(true_values==predictions)
    
    def fit(self, X, y): 
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats 
        """                        
        nSamples, nFeatures = X.shape #int to define number of samples of a feature

        self.weights = np.zeros(nFeatures) #Holds value for weights(slopes) of every feature
        self.bias = 0

        for epoch in range (self.epochs): #For every time algorithm must refine value, set by epochs 
            y_pred = self.sigmoid_function(self.lin_model(X))
            grad_w, grad_b = self.compute_gradients(X,y,y_pred,nSamples)
            self.update_parameters(grad_w,grad_b)

            loss = self._compute_loss(y,y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y,pred_to_class))
            self.losses.append(loss)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        lin_model = np.matmul(self.weights, X.transpose()) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
    
    def predict_proba(self, X):
        lin_model = np.matmul(self.weights, X.transpose()) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return y_pred 

