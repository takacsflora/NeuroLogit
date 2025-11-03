import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_base_scipy import Logit,MultinomialLogit

class av_pseudoPlotter():
    def __init__(self):
        # get dense points for vis
        nPredPoints = 600
        Vmodel = np.linspace(-1, 1, nPredPoints)
        # gamma transform
        visDiff_pseudo = np.sign(Vmodel) * np.abs(Vmodel)  # **gamma
        x_ = visDiff_pseudo
        # get samples for auditory
        As = np.array([-1, 0, 1]).astype("float")  # normally L/C/R so hardcode
        audDiff_pseudo = np.ones(nPredPoints) * As[:, np.newaxis]

        all_pseudo = []
        for a in audDiff_pseudo:
            # re-create matrix with what we are predicting
            pseudo_trials = pd.DataFrame()
            pseudo_trials["visL"] = np.abs(visDiff_pseudo) * (visDiff_pseudo < 0)
            pseudo_trials["visR"] = np.abs(visDiff_pseudo) * (visDiff_pseudo > 0)

            pseudo_trials["audL"] = np.abs(a) * (a < 0)
            pseudo_trials["audR"] = np.abs(a) * (a > 0)
            all_pseudo.append(pseudo_trials)

        self.pseudo = all_pseudo

    def update_pseudo(self, extra_predictors=None):
        """
        this function allows to update the pseudo in a dynamically so that the pseudo-matrix can be plugged directly into the prediction function. For example add baseline==1
        etc. importantly it takes only one value, which is on purpose. If you are plottign with a covariate, please plot separate predictions.
        
        """
        if extra_predictors is not None:
            for key, value in extra_predictors.items():
                for df in self.pseudo:
                    df[key] = value


    def plot_pseudo_predictions(
        self, gamma = 1,yscale="log", ax=None, predplotkwargs={"ls": "-"}, **kwargs
    ):
        assert hasattr(self, 'pseudo'), 'pseudo trials have not been generated'
                
        assert hasattr(self, 'predict_proba'), 'way to predict is required for this func to work'


        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

        self.update_pseudo(**kwargs)

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(self.pseudo)))

        for a, mycolor in zip(self.pseudo, colors):
            x_ = a["visR"] ** gamma - a["visL"] ** gamma
            y_ = self.predict_proba(a)

            if yscale == "log":
                y_ = np.log(y_ / (1 - y_))

            ax.plot(x_, y_, color=mycolor, **predplotkwargs)
# the av_split model which can have some opto permutations (all rely on parameters audL, audR, visL, visR, bias)
    
class av_split(Logit,av_pseudoPlotter):
    def __init__(self,extra_param_names = None,extra_param_init = None, extra_param_bounds = None):
        # Define the parameter names and initial values
        param_names = ['audR', 'audL', 'visR', 'visL', 'gamma', 'bias']
        param_init = {
            'bias': 0
        }

        param_bounds = {
            "gamma": (0.2, 2),
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
        av_pseudoPlotter.__init__(self)

    def predict_log_proba(self, X):
        self.check_params()  # Ensure all params are initialized

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        # Calculate log odds
        log_odds = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['bias']
        )
        return log_odds

class av_opto(av_split):
    def __init__(self):
       extra_param_names = ['audR_opto','audL_opto','visR_opto','visL_opto','bias_opto']
       extra_param_init = {
           'audR_opto': 1,
           'audL_opto': 1,
           'visR_opto': 1,
           'visL_opto': 1,
           'bias_opto': 0
       }

       super().__init__(extra_param_names,extra_param_init)
    
    def predict_log_proba(self, X):
        
        self.check_params()  # Ensure all params are initialized

        # Extract inputs
        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        O  = X[["bias_opto"]].values 


        # Calculate log odds
        control_odds = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['bias']
        )

        opto_odds = (
            self.params['visL_opto'] * vL * O +
            self.params['visR_opto'] * vR * O +
            self.params['audL_opto'] * aL * O +
            self.params['audR_opto'] * aR * O +
            self.params['bias_opto']        
        )


        return control_odds + opto_odds

    def update_pseudo(self, opto=1):
        pseudo = self.pseudo.copy()

        new = []
        for a in pseudo:
            a["bias_opto"] = opto
            new.append(a)

        self.pseudo = new   

class av_opto_hemispheric_additive(av_split):
    def __init__(self):
       extra_param_names = ['audR_opto','audL_opto','visR_opto','visL_opto','biasL_opto','biasR_opto']
       extra_param_init = {
           'audR_opto': 0,
           'audL_opto': 0,
           'visR_opto': 0,
           'visL_opto': 0,
           'biasR_opto': 0, 
           'biasL_opto': 0

       }

       extra_param_bounds = {
           'audR_opto': (None,0),
           'audL_opto': (0,None),
           'visR_opto': (None,0),
           'visL_opto': (0,None),          
       }

       super().__init__(extra_param_names,extra_param_init)

    def predict_log_proba(self, X):
        self.check_params()


        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")[:, np.newaxis]
        oL = (X["hemisphere"].values <= 0).astype("float")[:, np.newaxis] 


        # Calculate log odds
        control_odds = (
            self.params['visL'] * vL +
            self.params['visR'] * vR +
            self.params['audL'] * aL + 
            self.params['audR'] * aR +
            self.params['bias']
        )

        opto_odds = (
            self.params['visL_opto'] * vL * oR +
            self.params['visR_opto'] * vR * oL +
            self.params['audL_opto'] * aL * oR +
            self.params['audR_opto'] * aR * oL +
            self.params['biasL_opto']  * oR - 
            self.params['biasR_opto'] * oL      
        )


        return control_odds + opto_odds

    def update_pseudo(self,hemisphere=1):
        pseudo = self.pseudo.copy()

        new = []
        for a in pseudo:
            a["hemisphere"] = hemisphere
            new.append(a)

        self.pseudo = new   
# the divisive model which instead has a biasL/R parameter, but the opto acts on all parameters simultaneously
class av_opto_hemispheric_divisive(Logit,av_pseudoPlotter):
    def __init__(self):
        param_names = ['audR', 'audL', 'visR', 'visL', 'gamma', 'biasL','biasR','optoL','optoR']
        param_init = {
            'biasL': 0,
            'biasR': 0, 
            'optoL': 0,
            'optoR': 0
        }

        param_bounds = {
            "gamma": (0.2, 2),
            'optoL': (0,None),
            'optoR': (0,None)
        }

        super().__init__(param_names,param_init,param_bounds)
        av_pseudoPlotter.__init__(self)

    def predict_log_proba(self, X):
        self.check_params()


        vL = X[["visL"]].values ** self.params['gamma']
        vR = X[["visR"]].values ** self.params['gamma']
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")[:, np.newaxis]
        oL = (X["hemisphere"].values <= 0).astype("float")[:, np.newaxis] 

        right_odds = (
            (self.params['visR']* vR + 
             self.params['audR'] * aR + 
             self.params['biasR']) / (1+self.params['optoL']*oL)
         )
        
        left_odds = (
            (self.params['visL']* vL + 
             self.params['audL'] * aL + 
             self.params['biasL']) / (1+self.params['optoR']*oR)
         )
        
        return right_odds + left_odds
        
    def update_pseudo(self,hemisphere=1):
        pseudo = self.pseudo.copy()

        new = []
        for a in pseudo:
            a["hemisphere"] = hemisphere
            new.append(a)

        self.pseudo = new
