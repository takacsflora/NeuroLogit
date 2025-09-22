
# multiclass models to fit nogos and assess the effects of optogenetics 
import numpy as np
from .model_base_scipy import MultinomialLogit


class multi_bias_only(MultinomialLogit):
    def __init__(self,extra_param_names = None,extra_param_init = None, extra_param_bounds = None):
        param_names = ['biasL','biasR']
        param_init = {
            'biasL': 0,
            'biasR': 0
        }

        param_bounds = {}
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
        aL = X[["audL"]].values
        ones_for_bias = np.ones_like(aL)
        
        zR = (
            ones_for_bias*self.params['biasR']
             )
        
        zL = (

            ones_for_bias*self.params['biasL']
                )

        return zL,zR 

class av_multi(MultinomialLogit): 
    """th base class similar to the one emplyed by Zaatka-Haas to fit multinomal for nogos, as well as left and right choices.
    Evidence for right cocies only comes from the right side (visR,audR) and vice versa for left choices.

    Args:
        MultinomialLogit (_type_): fitter function that minimised the logLoss for the multinomial model
    """


    def __init__(self,extra_param_names = None,extra_param_init = None, extra_param_bounds = None):
        param_names = ['audR', 'audL', 'visR', 'visL', 'gamma', 'biasL','biasR']
        param_init = {
            'biasL': 0,
            'biasR': 0
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

        zR = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            self.params['biasR']
             )
        
        zL = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            self.params['biasL']
                )

        return zL,zR 

class av_multi_symmetric(av_multi):
    """
    the same as the av_multi model but the evidence for right choices comes from both the right and left side and so on
    """
    def __init__(self):
        
        # Call parent class's init to handle the parameter setup
        super().__init__()

    def predict_log_proba(self, X):
        self.check_params()  # Ensure all params are initialized

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        zR = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            -self.params['visL'] * vL +
            -self.params['audL'] * aL +
            self.params['biasR']
             )
        
        zL = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            -self.params['visR'] * vR +
            -self.params['audR'] * aR +
            self.params['biasL']
                )

        return zL,zR 

class av_multi_symmetric_audio(av_multi):
    """
    the same as the av_multi model but the evidence for right choices comes from both the right and left side for audio
    """
    def __init__(self):
        
        # Call parent class's init to handle the parameter setup
        super().__init__()

    def predict_log_proba(self, X):
        self.check_params()  # Ensure all params are initialized

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        zR = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            -self.params['audL'] * aL +
            self.params['biasR']
             )
        
        zL = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            -self.params['audR'] * aR +
            self.params['biasL']
                )

        return zL,zR  
## the asymetric models where the parameters can also take new values depedning on which side they support
class av_multi_asymetric_audio(av_multi):
    def __init__(self,
                 extra_param_names = ['audR_onL','audL_onR'],
                 extra_param_init = {'audR_onL': 1,'audL_onR': 1}):

       super().__init__(extra_param_names,extra_param_init)

    def predict_log_proba(self, X):
        self.check_params()

            # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        zR = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            self.params['audL_onR'] * aL +
            self.params['biasR']
             )
        
        zL = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            self.params['audR_onL'] * aR +
            self.params['biasL']
                )


        return zL,zR   

class av_multi_asymetric(av_multi): 
    def __init__(self,
                 extra_param_names = ['audR_onL','audL_onR','visR_onL','visL_onR'],
                 extra_param_init = {'audR_onL': 1,'audL_onR': 1,'visR_onL': 1,'visL_onR': 1}):

       super().__init__(extra_param_names,extra_param_init)

    def predict_log_proba(self, X):
        self.check_params()

            # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        zR = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            self.params['visL_onR'] * vL +
            self.params['audL_onR'] * aL +
            self.params['biasR']
             )
        
        zL = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            self.params['visR_onL'] * vR +
            self.params['audR_onL'] * aR +
            self.params['biasL']
                )
        


        return zL,zR       


#### the opto models  ####

# maybe chainge to the symmetric audio model
class avm_opto(av_multi):
    def __init__(self,
                 extra_param_names = [
                        'visR_opto','visL_opto','audR_opto','audL_opto',
                        'biasR_opto','biasL_opto'
                        ],
                 extra_param_init = {
                        'visR_opto': 0,
                        'visL_opto': 0,
                        'audR_opto': 0,
                        'audL_opto': 0,
                        'biasR_opto': 0,
                        'biasL_opto': 0                        
                     }):

       super().__init__(extra_param_names,extra_param_init)

    def predict_log_proba(self, X):
        self.check_params()

            # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        o = X[["bias_opto"]].values


        zR_ctrl = (
            self.params['visR'] * vR +
            self.params['audR'] * aR +
            -self.params['audL'] * aL +
            self.params['biasR']
             )
        
        zR_opto = (
            self.params['visR_opto'] * vR * o +
            self.params['audR_opto'] * aR  * o +
            -self.params['audL_opto'] * aL * o +
            self.params['biasR_opto'] * o 
             )
        
        zL_ctrl = (
            self.params['visL'] * vL +
            self.params['audL'] * aL +
            -self.params['audR'] * aR +
            self.params['biasL']
                )
        
        zL_opto = (
            self.params['visL_opto'] * vL * o  +
            self.params['audL_opto'] * aL * o +
            -self.params['audR_opto'] * aR * o +
            self.params['biasL_opto'] * o 
                )

        return zL_ctrl-zL_opto,zR_ctrl-zR_opto    
