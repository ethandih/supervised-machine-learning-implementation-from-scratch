import numpy as np

class LinearRegression():
    
    def __init__(self, L=0.001,epochs=3000): 
        self.L = L
        self.epochs = epochs
        self.losses, self.train_accuracies = [], []
        self.weights, self.bias = None, None #private: vector<float> weights;
        pass

    def precision(self,y,y_pred):
        return np.mean(y-y_pred)**2

        
    def fit(self, X, y): 
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats 
        """                        
        nSamples = len(X) #int to define number of samples of a feature



        if len(X.shape) == 1: #If only one feature, array is of shape (nSamples,) and not (nSamples,1)
            X = np.reshape(X.to_numpy(),newshape=(nSamples,1))
        else:
            X = np.reshape(X.to_numpy(),newshape=(nSamples,X.shape[1]))
        nFeatures = X.shape[1]

        slope = 0
        constantledd = 0
        self.weights = np.zeros((nFeatures,2)) #Holds values for slope and const ledd for every feature

        for feature in range ((nFeatures)): #For every feature, find slope and y intercept
            for j in range (self.epochs): #For every time algorithm must refine value, set by epochs 
                Y_pred = slope*X[0:,feature] + constantledd
                D_slope = (-2/nSamples) * np.sum(X[0:,feature]*(y-Y_pred))
                D_c = (-2/nSamples) * np.sum(y-Y_pred)
                slope -= D_slope*self.L
                constantledd -= D_c*self.L
                self.losses.append(self.precision(Y_pred,y))
            self.weights[feature] = np.array([slope, constantledd]) #Algorithm completed for one feature, updating weights output array
    
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
        if len(X.shape) == 1: #If only one feature, array is of shape (nSamples,) and not (nSamples,1)
            X = np.reshape(X.to_numpy(),newshape=(len(X),1))
        nFeatures = X.shape[1]
        nTestSamples = len(X)
        y_predictions = np.zeros((nTestSamples, nFeatures))
        for feature in range (nFeatures):
            for n in range(nTestSamples):
                y_predictions[n][feature] = self.weights[feature][0]*X[n][feature] + self.weights[feature][1]
        return y_predictions

