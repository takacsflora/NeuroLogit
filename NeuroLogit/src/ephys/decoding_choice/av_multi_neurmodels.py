
import numpy as np
from src.models.model_base_scipy import MultinomialLogit


class av_multi_neural_unilateral(MultinomialLogit): 
    """th base class similar to the one emplyed by Zaatka-Haas to fit multinomal for nogos, as well as left and right choices.
    Evidence for right cocies only comes from the right side (visR,audR) and vice versa for left choices.

    Args:
        MultinomialLogit (_type_): fitter function that minimised the logLoss for the multinomial model
    """


    def __init__(self,extra_param_names = None,extra_param_init = None, extra_param_bounds = None):
        param_names = ['audR', 'audL', 'visR', 'visL', 'gamma','biasR','biasL']
        param_init = {
            'audR': 1.0,
            'audL': 1.0,
            'visR': 1.0,
            'visL': 1.0, 
            'biasR': 0.0,
            'biasL': 0.0,
        }


        param_bounds = {
            "gamma": (0.2, 2),
        }
        # update with the added extras
        if extra_param_names is not None:
            param_names += extra_param_names        
        if extra_param_init is not None:
            param_init.update(extra_param_init) 
        if extra_param_bounds is not None:
            param_bounds.update(extra_param_bounds)
        
        # Call parent class's init to handle the parameter setup
        super().__init__(param_names, param_init,param_bounds)

    def predict_log_proba(self, X):
        self.check_params()  # Ensure all params are initialized

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        neural_columns_left_hemisphere = [col for col in X.columns if (('neuron_' in col) or ('pca' in col)) and col.endswith('_left')]
        neural_columns_right_hemisphere = [col for col in X.columns if (('neuron_' in col) or ('pca' in col)) and col.endswith('_right')]
        neural_activity_left = X[neural_columns_left_hemisphere].values
        neural_activity_right = X[neural_columns_right_hemisphere].values

        neural_weights_left_contra = np.array([self.params[f"{col}_on_contra"] for col in neural_columns_left_hemisphere])
        neural_weights_right_contra = np.array([self.params[f"{col}_on_contra"] for col in neural_columns_right_hemisphere])
        neural_weights_left_ipsi = np.array([self.params[f"{col}_on_ipsi"] for col in neural_columns_left_hemisphere])
        neural_weights_right_ipsi = np.array([self.params[f"{col}_on_ipsi"] for col in neural_columns_right_hemisphere])

        neural_contrib_left_on_contra = neural_activity_left @ neural_weights_left_contra[:, np.newaxis]
        neural_contrib_left_on_ipsi = neural_activity_left @ neural_weights_left_ipsi[:, np.newaxis]
  
        neural_contrib_right_on_contra = neural_activity_right @ neural_weights_right_contra[:, np.newaxis]
        neural_contrib_right_on_ipsi = neural_activity_right @ neural_weights_right_ipsi[:, np.newaxis]


        zR = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            -self.params['audL'] * aL +
            neural_contrib_left_on_contra + 
            neural_contrib_right_on_ipsi +
            self.params['biasR']
        )
        
        zL = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            -self.params['audR'] * aR +  
            neural_contrib_right_on_contra + 
            neural_contrib_left_on_ipsi +
            self.params['biasL']
        )

        return zL,zR 
    

