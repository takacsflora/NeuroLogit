# %% 

# I decided to also benchmark scipy...
import numpy as np
import pandas as pd 

from scipy.optimize import minimize
from sklearn.metrics import log_loss

#%%
class AV_opto_model():
    def __init__(self):
        # Initialize parameters
        self.audR = 1
        self.audL = 1 
        self.visR = 1 
        self.visL = 1 

        # Opto parameters
        self.audR_opto = 0
        self.audL_opto = 0 
        self.visR_opto = 0 
        self.visL_opto = 0 

        # Extra parameters
        self.gamma = 1  # This can be fixed during training
        self.bias = 0 
        self.bias_opto = 0

    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[['visL']].values
        vR = X[['visR']].values 
        aL = X[['audL']].values
        aR = X[['audR']].values
        O = X[['bias_opto']].values

        # Power visual parameters to gamma
        vL_ = vL ** self.gamma
        vR_ = vR ** self.gamma

        # Compute signal without opto
        S = (self.visL * vL_ +
             self.visR * vR_ +
             self.audL * aL +
             self.audR * aR + 
             self.bias)

        # Compute signal with opto
        S_opto = (self.visL_opto * vL_ * O +
                  self.visR_opto * vR_ * O +
                  self.audL_opto * aL * O +
                  self.audR_opto * aR * O + 
                  self.bias_opto)

        # Calculate logits for probabilities
        logOdds = S + S_opto

        return logOdds

    @staticmethod
    def raise_to_sigm(logOdds):
        'function to calculate pR from logOdds'
        return np.exp(logOdds) / (1 + np.exp(logOdds))

    def predict_proba(self, X):
        logOdds = self.predict_log_proba(X)
        return self.raise_to_sigm(logOdds)


    def predict(self,X):
        logOdds = self.predict_log_proba(X)    
        choices = (np.sign(logOdds)>0).astype('float')
        return choices
    
    def decision_function(self,X):
        pass

    def get_params(self):
        # Ensure all values are standard floats
        return {
            'audR': float(self.audR),
            'audL': float(self.audL),
            'visR': float(self.visR),
            'visL': float(self.visL),
            'audR_opto': float(self.audR_opto),
            'audL_opto': float(self.audL_opto),
            'visR_opto': float(self.visR_opto),
            'visL_opto': float(self.visL_opto),
            'gamma': float(self.gamma),
            'bias': float(self.bias),
            'bias_opto': float(self.bias_opto)
        }
    

    def set_params(self, params):
        # Ensure all values are standard floats when setting parameters
        for key, value in params.items():
            setattr(self, key, float(value))  # Convert to standard float

    def _objective(self, params, X, y, param_mask, fixed_params):
        """
        Objective function to minimize (e.g., negative log-likelihood or squared loss).
        params: list of trainable parameters
        param_mask: list indicating which parameters are being trained (1 = trainable, 0 = fixed)
        fixed_params: the fixed parameters
        """
        # Map trainable parameters to the full set
        full_params = fixed_params.copy()
        full_params.update({k: p for k, p in zip(param_mask.keys(), params)})

        # Set the full parameters in the model
        self.set_params(full_params)

        # Get predictions and calculate loss
        predictions = self.predict_proba(X) # phat for each
        
        loss = log_loss(y,predictions,normalize=True)

        return loss

    def fit(self, X, y, fixed_params={}):
        """
        Fit the model using scipy.optimize.minimize.
        X: input features
        y: target labels
        fixed_params: dictionary of fixed parameters (won't be optimized)
        """
        # Get all model parameters
        all_params = self.get_params()

        # Separate trainable and fixed parameters
        param_mask = {k: v for k, v in all_params.items() if k not in fixed_params}
        fixed_params = {k: v for k, v in all_params.items() if k in fixed_params}

        # Convert to lists for minimize
        trainable_params = list(param_mask.values())
        trainable_keys = list(param_mask.keys())

        # Define bounds (if needed)
        bounds = [(0.2, 2) if key == 'gamma' else (None, None) for key in trainable_keys]
        # Run optimization
        result = minimize(self._objective, trainable_params, args=(X, y, param_mask, fixed_params), bounds=bounds)

        # Set the optimized parameters in the model
        optimized_params = {k: v for k, v in zip(trainable_keys, result.x)}
        self.set_params(optimized_params)

        return result

    def score(self, X, y):
        """
        Score the model (e.g., accuracy or log-loss).
        """
        predictions = self.predict_proba(X)
        accuracy = np.mean((predictions > 0.5) == y)
        return accuracy