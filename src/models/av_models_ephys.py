# I haven't tested this one yet actually.
import numpy as np
from .model_base_scipy import LinearRegression
from .av_models_opto import av_pseudoPlotter # might need to be organised into a subprocess

## basic classes
class baseline(LinearRegression):
    def __init__(self, extra_param_names=None, extra_param_init=None, extra_param_bounds=None):
        # Define the parameter names and initial values
        param_names = ['baseline']
        param_init = {
            'baseline': 0
        }

        param_bounds = {}

        # update with the added extras
        if extra_param_names is not None:
            param_names = param_names + extra_param_names        
        if extra_param_init is not None:
            param_init.update(extra_param_init) 
        if extra_param_bounds is not None:
            param_bounds.update(extra_param_bounds)
        
        # Call parent class's init to handle the parameter setup
        super().__init__(param_names, param_init,param_bounds)
        self.param_names = param_names
        self.param_init = param_init
        self.param_bounds = param_bounds

    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Create predictions for each row in X
        additive_pred = (
            np.full(X.shape[0], self.params['baseline'])
        )
        return additive_pred

class vis(LinearRegression):
    
    def __init__(self, extra_param_names=None, extra_param_init=None, extra_param_bounds=None):
        # Define the parameter names and initial values
        param_names = ['visL','visR','gamma','baseline']
        param_init = {
            'visR': 1,
            'visL': 1,
            'gamma': 1, 
            'baseline': 0
        }
            
            
        param_bounds = {
                "gamma": (0.001, 5),
            }


        # update with the added extras
        if extra_param_names is not None:
            param_names = param_names + extra_param_names        
        if extra_param_init is not None:
            param_init.update(extra_param_init) 
        if extra_param_bounds is not None:
            param_bounds.update(extra_param_bounds)
        
        # Call parent class's init to handle the parameter setup
        super().__init__(param_names, param_init,param_bounds)
        self.param_names = param_names
        self.param_init = param_init
        self.param_bounds = param_bounds
    
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']


        additive_pred = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['baseline']
        )
        return additive_pred
    
class aud(LinearRegression):

    def __init__(self, extra_param_names=None, extra_param_init=None, extra_param_bounds=None):
        # Define the parameter names and initial values
        param_names = ['audL', 'audR','baseline']
        param_init =  {
            'audR': 1,
            'audL': 1, 
            'baseline': 0
        }
            
        param_bounds = {}


        # update with the added extras
        if extra_param_names is not None:
            param_names = param_names + extra_param_names        
        if extra_param_init is not None:
            param_init.update(extra_param_init) 
        if extra_param_bounds is not None:
            param_bounds.update(extra_param_bounds)
        
        # Call parent class's init to handle the parameter setup
        super().__init__(param_names, param_init,param_bounds)
        self.param_names = param_names
        self.param_init = param_init
        self.param_bounds = param_bounds

    
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        additive_pred = (
            self.params['audL'] * aL +
            self.params['audR'] * aR +
            self.params['baseline']
        )
        return additive_pred

class audiovisual(vis):
    def __init__(self):
        extra_param_names = ['audL', 'audR']
        extra_param_init = {
            'audR': 1,
            'audL': 1
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)
    
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        # Calculate log odds
        additive_pred = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['baseline']
        )

        return additive_pred

###
# gain models where the slope 
class vis_engagement_gain(vis): 
    def __init__(self):
        extra_param_names = ['visL_active', 'visR_active','baseline_active']
        extra_param_init = {
            'visL_active': 0,
            'visR_active': 0,
            'baseline_active': 0

        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)
    
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        active = X[["is_active"]].values

        passive_resps = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['visL_active'] * vL * active +
            self.params['visR_active'] * vR * active +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred
    
class aud_engagement_gain(aud): 
    def __init__(self):
        extra_param_names = ['audL_active', 'audR_active','baseline_active']
        extra_param_init = {
            'audL_active': 0,
            'audR_active': 0,
            'baseline_active': 0

        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X):  
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        active = X[["is_active"]].values

        passive_resps = (
            self.params['audL'] * aL +
            self.params['audR'] * aR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['audL_active'] * aL * active +
            self.params['audR_active'] * aR * active +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred    
    
class audiovisual_engagement_gain(vis):
    def __init__(self):
        extra_param_names = ['audL', 'audR','visL_active', 'visR_active','audL_active', 'audR_active','baseline_active']
        extra_param_init = {
            'audR': 1,
            'audL': 1,
            'visL_active': 0,
            'visR_active': 0,
            'audL_active': 0,
            'audR_active': 0,
            'baseline_active': 0

        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)#

    def predict(self, X): 
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        active = X[["is_active"]].values

        passive_resps = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['visL_active'] * vL * active +
            self.params['visR_active'] * vR * active +
            self.params['audL_active'] * aL * active +
            self.params['audR_active'] * aR * active +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

class baseline_engagement(baseline):
    def __init__(self):
        extra_param_names = ['baseline_active']
        extra_param_init = {
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)#

    def predict(self, X): 
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        active = X[["is_active"]].values

        passive_resps = (
            self.params['baseline']
        )

        active_modulation = (
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

class vis_engagement(vis):
    def __init__(self):
        extra_param_names = ['baseline_active']
        extra_param_init = {
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)
        
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        active = X[["is_active"]].values

        passive_resps = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred
   
# need to refine the engagement models too, i.e. just baseline engagement etc. 
class aud_engagement(aud):
    def __init__(self):
        extra_param_names = ['baseline_active']
        extra_param_init = {
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X):  
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        active = X[["is_active"]].values

        passive_resps = (
            self.params['audL'] * aL +
            self.params['audR'] * aR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred

class audiovisual_engagement(vis):
    def __init__(self):
        extra_param_names = ['audL', 'audR','baseline_active']
        extra_param_init = {
            'audR': 1,
            'audL': 1,
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X): 
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        active = X[["is_active"]].values

        passive_resps = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred
###
class choice(baseline):
    def __init__(self):
        extra_param_names = ['choice_left', 'choice_right']
        extra_param_init = {
            'choice_left': 0, 
            'choice_right': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)#

    def predict(self, X): 
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int')


        additive_pred = (
            self.params['choice_left'] * choice_left +
            self.params['choice_right'] * choice_right +
            self.params['baseline']
        )

        return additive_pred

class vis_choice(vis): 
    def __init__(self):
        extra_param_names = ['choice_left', 'choice_right']
        extra_param_init = {
            'choice_left': 0, 
            'choice_right': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X):
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int')

        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']


        additive_pred = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +

            self.params['choice_left'] * choice_left +
            self.params['choice_right'] * choice_right +
            self.params['baseline']
        )

        return additive_pred

class aud_choice(aud): 
    def __init__(self):
        extra_param_names = ['choice_left', 'choice_right']
        extra_param_init = {
            'choice_left': 0, 
            'choice_right': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X):
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int')

        aL = X[["audL"]].values
        aR = X[["audR"]].values

        additive_pred = (
            self.params['audL'] * aL +
            self.params['audR'] * aR +

            self.params['choice_left'] * choice_left +
            self.params['choice_right'] * choice_right +
            self.params['baseline']
        )

        return additive_pred

class audiovisual_choice(vis): 
    def __init__(self):
        extra_param_names = ['audL', 'audR','choice_left', 'choice_right']
        extra_param_init = {
            'audR': 1,
            'audL': 1,
            'choice_left': 0, 
            'choice_right': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int')

        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        additive_pred = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +

            self.params['choice_left'] * choice_left +
            self.params['choice_right'] * choice_right +
            self.params['baseline']
        )

        return additive_pred

class baseline_choice(baseline): 
    def __init__(self):
        extra_param_names = ['choice_left', 'choice_right']
        extra_param_init = {
            'choice_left': 0, 
            'choice_right': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):  
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int')

        additive_pred = (
            self.params['choice_left'] * choice_left +
            self.params['choice_right'] * choice_right +
            self.params['baseline']
        )

        return additive_pred


# engagemen + choice model combined
class baseline_choice_engaged(baseline): 
    def __init__(self):
        extra_param_names = ['choice','baseline_active']
        extra_param_init = {
            'choice': 0, 
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):  
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int') * -1 

        choice = choice_left + choice_right

        active = X[["is_active"]].values

        passive_resps = (

            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

class aud_choice_engaged(aud): 
    def __init__(self):
        extra_param_names = ['choice','baseline_active']
        extra_param_init = {
            'choice': 0, 
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X): 
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int') * -1 

        choice = choice_left + choice_right

        aL = X[["audL"]].values
        aR = X[["audR"]].values

        active = X[["is_active"]].values

        passive_resps = (
            self.params['audL'] * aL +
            self.params['audR'] * aR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

class vis_choice_engaged(vis):
    def __init__(self):
        extra_param_names = ['choice','baseline_active']
        extra_param_init = {
            'choice': 0, 
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int') * -1 

        choice = choice_left + choice_right

        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']

        active = X[["is_active"]].values

        passive_resps = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred
    
class audiovisual_choice_engaged(vis):
    def __init__(self):
        extra_param_names = ['audL', 'audR','choice','baseline_active']
        extra_param_init = {
            'audR': 1,
            'audL': 1,
            'choice': 0, 
            'baseline_active': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):
        self.check_params()

        # Extract inputs
        choice_right = (X[["choice"]]==1).values.astype('int')
        choice_left = (X[["choice"]]==0).values.astype('int') * -1 

        choice = choice_left + choice_right

        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        active = X[["is_active"]].values

        passive_resps = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['baseline_active'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred  


### 

# also I think I will deploy all the functions to myriad in the end because it is a lot of models.

# will add a class that also take movement as a predictor for all of these models
# also will need to add a class that takes active/Passive as predictors
# and another class for choice. 
