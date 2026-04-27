# contains the generic Logit module build in scipy

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BaseTrainer(ABC):
    """
    Base class for a training models with scipy with the option to fix certain parameters and set the bounds.

    """
    def __init__(self, param_names, param_init=None, param_bounds=None):
        # Initialize parameters as a dictionary basically the inial parameter is 1 
        # and the initial bound is (None,None) unless you specify in a dictionary

        if param_init is None:
            param_init = {name: 1 for name in param_names}
        self.params = {name: param_init.get(name, 1) for name in param_names}
        
        if param_bounds is None:
            param_bounds = {name: (None, None) for name in param_names}
        self.bounds = {name: param_bounds.get(name, (None, None)) for name in param_names}

    def check_params(self):
        # Ensure all parameters are properly initialized
        for name, value in self.params.items():
            if value is None:
                raise ValueError(f"Parameter {name} has not been initialized.")
    
    def print_params(self, fixed_params=None):
        print("Model Parameters:")
        for name, value in self.params.items():
            bounds = self.bounds[name]
            fixed_status = " (fixed)" if fixed_params and name in fixed_params else ""
            print(f"{name}: Initial Value = {value}, Bounds = {bounds}{fixed_status}")

    def get_params(self):
        return self.params

    def get_bounds(self):
        return self.bounds
    
    def set_params(self, params):
        # Update parameters from a dictionary, ensuring valid keys
        for key, value in params.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise ValueError(f"Parameter {key} not recognized.")

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def _objective(self, params, X, y, param_mask, fixed_params):
        pass
    
    
    def fit(self, X, y, fixed_params={},verbose = False):
        """
        Fit the model using scipy.optimize.minimize.
        X: input features
        y: target labels
        fixed_params: dictionary of fixed parameters (won't be optimized)
        """
        self.check_params()  # Ensure all params are initialized
        # Get all model parameters
        all_params = self.get_params()
        all_bounds = self.get_bounds()

        # Separate trainable and fixed parameters
        param_mask = {k: v for k, v in all_params.items() if k not in fixed_params}

        # Convert to lists for minimize
        trainable_param_starting_points = list(param_mask.values())
        trainable_param_names = list(param_mask.keys())

        # Define bounds (if needed)
        trainable_param_bounds = [all_bounds[key] for key in trainable_param_names]

        if verbose:
            print('initialising with:')
            self.print_params()

        # Run optimization
        result = minimize(
            self._objective,
            trainable_param_starting_points,
            args=(X, y, param_mask, fixed_params),
            bounds=trainable_param_bounds,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'disp': False, 'gtol': 1e-6},  # Suppress printing all the output

        )

        if verbose:
            print(f'fitting result success: {result.success}')
            if result.success:
                print(self.params)

        # Set the optimized parameters in the model
        optimized_params = {k: v for k, v in zip(trainable_param_names, result.x)}
        
        self.set_params(optimized_params)

    def score(self, X, y,scorer = 'log_loss',**scorerkwrargs):
        """
        Score the model (e.g., accuracy or log-loss).
        """
        y_pred = self.predict_proba(X)

        scorer_func = getattr(metrics,scorer)

        score = scorer_func(y,y_pred,**scorerkwrargs)

        return score 

class Logit(BaseTrainer):
        # Call parent class's init to handle the parameter setup
    def __init__(self, param_names, param_init=None, param_bounds=None):
        super().__init__(param_names, param_init,param_bounds)

    @staticmethod
    def raise_to_sigm(logOdds):
        "function to calculate pR from logOdds"
        return np.exp(logOdds) / (1 + np.exp(logOdds))
    
    @abstractmethod
    def predict_log_proba(self, X):
        pass

    def predict_proba(self, X):
        return self.raise_to_sigm(self.predict_log_proba(X))

    def predict(self, X):
        return (self.predict_log_proba(X) > 0).astype('float')
    
    def _objective(self, params, X, y, param_mask, fixed_params):
        """
        Objective function to minimize (e.g., negative log-likelihood or squared loss).
        params: list of trainable parameters
        param_mask: list indicating which parameters are being trained (1 = trainable, 0 = fixed)
        fixed_params: the fixed parameters
        """
        # Map trainable parameters to the full set
        full_params = {**fixed_params, **dict(zip(param_mask.keys(), params))}
        self.set_params(full_params)

        # Get predictions and calculate loss
        predictions = self.predict_proba(X) 
        return metrics.log_loss(y, predictions, normalize=True) 
    
class MultinomialLogit(Logit):
    def __init__(self, param_names, param_init=None, param_bounds=None):
        super().__init__(param_names, param_init,param_bounds)

    @abstractmethod
    def predict_log_proba(self, X):
        """
        this fucntion now needs to return two numbers 
        zL (logIdds of left vs Nogo) and zR (logIdds of right vs Nogo)
        """
        pass

    
    def predict_proba(self, X):
        zL,zR = self.predict_log_proba(X)

        # compute probabilities using the softmax function
        # expzL = np.exp(zL)
        # expzR = np.exp(zR)

        # denom = 1 + expzL + expzR
        # pNoGo = 1 / denom
        # pL = expzL / denom
        # pR = expzR / denom

        # stable softmax computation
        maxz = np.maximum.reduce([zL, zR, np.zeros_like(zL)])

        exp0 = np.exp(0 - maxz)
        expL = np.exp(zL - maxz)
        expR = np.exp(zR - maxz)
        denom = exp0 + expL + expR
        pNoGo = exp0 / denom
        pL = expL / denom
        pR = expR / denom

        return np.hstack([pNoGo, pL, pR])
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class LinearRegression(BaseTrainer):
    def __init__(self, param_names, param_init=None, param_bounds=None):
        super().__init__(param_names, param_init,param_bounds)

    def predict(self, X):
        return np.dot(X, np.array(list(self.params.values())))

    def _objective(self, params, X, y, param_mask, fixed_params):
        """
        Objective function to minimize (e.g., negative log-likelihood or squared loss).
        params: list of trainable parameters
        param_mask: list indicating which parameters are being trained (1 = trainable, 0 = fixed)
        fixed_params: the fixed parameters
        """
        # Map trainable parameters to the full set
        full_params = {**fixed_params, **dict(zip(param_mask.keys(), params))}
        self.set_params(full_params)

        # Get predictions and calculate loss
        predictions = self.predict(X) 
        return metrics.mean_squared_error(y, predictions)