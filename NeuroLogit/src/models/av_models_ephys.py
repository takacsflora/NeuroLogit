# I haven't tested this one yet actually.
import numpy as np
from .model_base_scipy import LinearRegression
from .av_models_opto import av_pseudoPlotter # might need to be organised into a subprocess

## basic classes

###### HELPERS ######
##### heper function to convert left rigth to ipsi and contra relative to neuron
def get_ipsi_contra(X,which='vis'):
    if which == 'vis':
        visDiff = (X['visR'] - X['visL']) * X['hemi']
        contra = visDiff.clip(lower=0)
        ipsi = np.abs(visDiff.clip(upper=0))

    elif which == 'aud':
        audDiff = (X['audR'] - X['audL']) * X['hemi']
        contra = audDiff.clip(lower=0)
        ipsi = np.abs(audDiff.clip(upper=0))

    elif which == 'choice':
        choice_right = (X["choice"]==1).values.astype('int')
        choice_left = (X["choice"]==0).values.astype('int') * -1 

        choice = (choice_left + choice_right) * X['hemi']

        ipsi = choice 
        contra = np.nan

    return ipsi,contra

def get_param_maths_symbols():
    return {
        'visC': r'V_{C}',
        'visI': r'V_{I}',
        'audC': r'A_{C}',
        'audI': r'A_{I}',
        'baseline': r'k',
        'task': r'k_{active}',
        'visC_gain': r'V_{C}^{gain}',
        'audC_gain': r'A_{C}^{gain}',
        'audI_gain': r'A_{I}^{gain}',
        'choice': r'k_{choice}',
    }

###### PASSIVE MODELS ######
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
        param_names = ['visC','gamma','baseline']
        param_init = {
            'visC': 1, # means vis contra 
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

        _, vC = get_ipsi_contra(X,which='vis')
        # Extract inputs
        vC = vC ** self.params['gamma']


        additive_pred = (
            self.params['visC'] * vC +
            self.params['baseline']
        )
        return additive_pred    

class aud(LinearRegression):

    def __init__(self, extra_param_names=None, extra_param_init=None, extra_param_bounds=None):
        # Define the parameter names and initial values
        param_names = ['audC', 'audI','baseline']
        param_init =  {
            'audC': 1,
            'audI': 1, 
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

        aI,aC = get_ipsi_contra(X,which='aud')

        additive_pred = (
            self.params['audC'] * aC +
            self.params['audI'] * aI +
            self.params['baseline']
        )
        return additive_pred

class audiovisual(vis):
    def __init__(self):
        extra_param_names = ['audC', 'audI']
        extra_param_init = {
            'audC': 1,
            'audI': 1
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)
    
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()


        _,vC = get_ipsi_contra(X,which='vis')
        aI,aC = get_ipsi_contra(X,which='aud')


        vC = vC ** self.params['gamma']
        # Calculate log odds
        additive_pred = (
            self.params['visC'] * vC +
            self.params['audC'] * aC + 
            self.params['audI'] * aI +
            self.params['baseline']
        )

        return additive_pred

###### ADDITIVE TASK MODULATION MODELS ###### (i.e. when a constant is added when the neuron engages in a task)

class baseline_task(baseline):
    def __init__(self):
        extra_param_names = ['task']
        extra_param_init = {
            'task': 0
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
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred
    
class vis_task(vis):
    def __init__(self):
        extra_param_names = ['task']
        extra_param_init = {
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)
        
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        _, vC = get_ipsi_contra(X,which='vis')
        # Extract inputs
        vC = vC ** self.params['gamma']

        active = X["is_active"].values

        passive_resps = (
            self.params['visC'] * vC +
            self.params['baseline']
        )

        active_modulation = (
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred
    
class aud_task(aud):
    def __init__(self):
        extra_param_names = ['task']
        extra_param_init = {
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X):  
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        aI,aC = get_ipsi_contra(X,which='aud')
        active = X["is_active"].values

        passive_resps = (
            self.params['audC'] * aC +
            self.params['audI'] * aI +
            self.params['baseline']
        )

        active_modulation = (
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred

class audiovisual_task(vis):
    def __init__(self):
        extra_param_names = ['audC', 'audI','task']
        extra_param_init = {
            'audR': 1,
            'audL': 1,
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X): 
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        _, vC = get_ipsi_contra(X,which='vis')
        vC = vC ** self.params['gamma']

        aI,aC = get_ipsi_contra(X,which='aud')
        active = X["is_active"].values

        passive_resps = (
            self.params['visC'] * vC +
            self.params['audC'] * aC + 
            self.params['audI'] * aI +
            self.params['baseline']
        )

        active_modulation = (
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

#### MULTIPLICATIVE TASK MODULATION MODELS ###### (i.e. when the tuning curve gets steeper/shallower when the neuron engages in a task) 

class vis_task_gain(vis): 
    def __init__(self): 
        extra_param_names = ['visC_gain','task']
        extra_param_init = {
            'visC_gain': 0,
            'task': 0

        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)
    
    def predict(self, X):
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        _, vC = get_ipsi_contra(X,which='vis')
        vC = vC ** self.params['gamma']
        active = X["is_active"].values

        passive_resps = (
            self.params['visC'] * vC +
            self.params['baseline']
        )

        active_modulation = (
            self.params['visC_gain'] * vC * active +
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred
    
class aud_task_gain(aud): 
    def __init__(self):
        extra_param_names = ['audC_gain', 'audI_gain','task']
        extra_param_init = {
            'audC_gain': 0,
            'audI_gain': 0,
            'task': 0

        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X):  
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        aI,aC = get_ipsi_contra(X,which='aud')
        active = X["is_active"].values

        passive_resps = (
            self.params['audC'] * aC +
            self.params['audI'] * aI +
            self.params['baseline']
        )

        active_modulation = (
            self.params['audC_gain'] * aC * active +
            self.params['audI_gain'] * aI * active +
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )
        return additive_pred    
    
class audiovisual_task_gain(vis):
    def __init__(self):
        extra_param_names = ['audC', 'audI','visC_gain','audC_gain', 'audI_gain','task']
        extra_param_init = {
            'audC': 1,
            'audI': 1,
            'visC_gain': 0,
            'audC_gain': 0,
            'audI_gain': 0,
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)#

    def predict(self, X): 
        # Ensure all params are initialized
        self.check_params()

        # Extract inputs
        _, vC = get_ipsi_contra(X,which='vis')
        vC = vC ** self.params['gamma']
        aI,aC = get_ipsi_contra(X,which='aud')
        active = X["is_active"].values

        passive_resps = (
            self.params['visC'] * vC + 
            self.params['audC'] * aC +
            self.params['audI'] * aI +
            self.params['baseline']
        )

        active_modulation = (
            self.params['visC_gain'] * vC * active +
            self.params['audC_gain'] * aC * active +
            self.params['audI_gain'] * aI * active +
            self.params['task'] * active
        )


        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

##### CHOICE MODELS ###### (i.e. when the neuron is more active when the animal chooses one side over the other)

class baseline_choice(baseline): 
    def __init__(self):
        extra_param_names = ['choice','task']
        extra_param_init = {
            'choice': 0, 
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):  
        self.check_params()

        # Extract inputs

        choice,_ = get_ipsi_contra(X,which='choice')
        active = X["is_active"].values

        passive_resps = (

            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

class aud_choice(aud): 
    def __init__(self):
        extra_param_names = ['choice','task']
        extra_param_init = {
            'choice': 0, 
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

    def predict(self, X): 
        self.check_params()

        choice,_ = get_ipsi_contra(X,which='choice')
        aI,aC = get_ipsi_contra(X,which='aud')

        active = X["is_active"].values

        passive_resps = (
            self.params['audC'] * aC +
            self.params['audI'] * aI +
            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred

class vis_choice(vis):
    def __init__(self):
        extra_param_names = ['choice','task']
        extra_param_init = {
            'choice': 0, 
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):
        self.check_params()

        choice,_ = get_ipsi_contra(X,which='choice')
        _,vC = get_ipsi_contra(X,which='vis')
        vC = vC ** self.params['gamma']

        active = X["is_active"].values

        passive_resps = (
            self.params['visC'] * vC +
            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred
    
class audiovisual_choice(vis):
    def __init__(self):
        extra_param_names = ['audC', 'audI','choice','task']
        extra_param_init = {
            'audR': 1,
            'audL': 1,
            'choice': 0, 
            'task': 0
        }
        
        extra_param_bounds = None

        super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

    def predict(self, X):
        self.check_params()

        choice,_ = get_ipsi_contra(X,which='choice')
        _,vC = get_ipsi_contra(X,which='vis')
        vC = vC ** self.params['gamma']
        aI,aC = get_ipsi_contra(X,which='aud')

        active = X["is_active"].values

        passive_resps = (
            self.params['visC'] * vC +
            self.params['audC'] * aC + 
            self.params['audI'] * aI +
            self.params['baseline']
        )

        active_modulation = (
            self.params['choice'] * choice +
            self.params['task'] * active      
        )

        additive_pred = (
            passive_resps + active_modulation
        )

        return additive_pred  



############################# OLD STUFF ###############################
# gain models where the slope 

   
# need to refine the engagement models too, i.e. just baseline engagement etc. 

### OLD MODELS THAT SPLIT TJE CHOICE INTO LEFT AND RIGHT ###
# class choice(baseline):
#     def __init__(self):
#         extra_param_names = ['choice_left', 'choice_right']
#         extra_param_init = {
#             'choice_left': 0, 
#             'choice_right': 0
#         }
        
#         extra_param_bounds = None

#         super().__init__(extra_param_names, extra_param_init, extra_param_bounds)#

#     def predict(self, X): 
#         # Ensure all params are initialized
#         self.check_params()

#         # Extract inputs
#         choice_right = (X[["choice"]]==1).values.astype('int')
#         choice_left = (X[["choice"]]==0).values.astype('int')


#         additive_pred = (
#             self.params['choice_left'] * choice_left +
#             self.params['choice_right'] * choice_right +
#             self.params['baseline']
#         )

#         return additive_pred

# class vis_choice(vis): 
#     def __init__(self):
#         extra_param_names = ['choice_left', 'choice_right']
#         extra_param_init = {
#             'choice_left': 0, 
#             'choice_right': 0
#         }
        
#         extra_param_bounds = None

#         super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

#     def predict(self, X):
#         self.check_params()

#         # Extract inputs
#         choice_right = (X[["choice"]]==1).values.astype('int')
#         choice_left = (X[["choice"]]==0).values.astype('int')

#         vL = X[["visL"]].values ** self.params['gamma']
#         vR = X[["visR"]].values ** self.params['gamma']


#         additive_pred = (
#             self.params['visL'] * vL +
#             self.params['visR'] * vR +

#             self.params['choice_left'] * choice_left +
#             self.params['choice_right'] * choice_right +
#             self.params['baseline']
#         )

#         return additive_pred

# class aud_choice(aud): 
#     def __init__(self):
#         extra_param_names = ['choice_left', 'choice_right']
#         extra_param_init = {
#             'choice_left': 0, 
#             'choice_right': 0
#         }
        
#         extra_param_bounds = None

#         super().__init__(extra_param_names, extra_param_init, extra_param_bounds)

#     def predict(self, X):
#         self.check_params()

#         # Extract inputs
#         choice_right = (X[["choice"]]==1).values.astype('int')
#         choice_left = (X[["choice"]]==0).values.astype('int')

#         aL = X[["audL"]].values
#         aR = X[["audR"]].values

#         additive_pred = (
#             self.params['audL'] * aL +
#             self.params['audR'] * aR +

#             self.params['choice_left'] * choice_left +
#             self.params['choice_right'] * choice_right +
#             self.params['baseline']
#         )

#         return additive_pred

# class audiovisual_choice(vis): 
#     def __init__(self):
#         extra_param_names = ['audL', 'audR','choice_left', 'choice_right']
#         extra_param_init = {
#             'audR': 1,
#             'audL': 1,
#             'choice_left': 0, 
#             'choice_right': 0
#         }
        
#         extra_param_bounds = None

#         super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

#     def predict(self, X):
#         self.check_params()

#         # Extract inputs
#         choice_right = (X[["choice"]]==1).values.astype('int')
#         choice_left = (X[["choice"]]==0).values.astype('int')

#         vL = X[["visL"]].values ** self.params['gamma']
#         vR = X[["visR"]].values ** self.params['gamma']
#         aL = X[["audL"]].values
#         aR = X[["audR"]].values

#         additive_pred = (
#             self.params['visL'] * vL +
#             self.params['visR'] * vR +
#             self.params['audL'] * aL + 
#             self.params['audR'] * aR +

#             self.params['choice_left'] * choice_left +
#             self.params['choice_right'] * choice_right +
#             self.params['baseline']
#         )

#         return additive_pred

# class baseline_choice(baseline): 
#     def __init__(self):
#         extra_param_names = ['choice_left', 'choice_right']
#         extra_param_init = {
#             'choice_left': 0, 
#             'choice_right': 0
#         }
        
#         extra_param_bounds = None

#         super().__init__(extra_param_names, extra_param_init, extra_param_bounds)  

#     def predict(self, X):  
#         self.check_params()

#         # Extract inputs
#         choice_right = (X[["choice"]]==1).values.astype('int')
#         choice_left = (X[["choice"]]==0).values.astype('int')

#         additive_pred = (
#             self.params['choice_left'] * choice_left +
#             self.params['choice_right'] * choice_right +
#             self.params['baseline']
#         )

#         return additive_pred






# engagemen + choice model combined


### 

# also I think I will deploy all the functions to myriad in the end because it is a lot of models.

# will add a class that also take movement as a predictor for all of these models
# also will need to add a class that takes active/Passive as predictors
# and another class for choice. 
