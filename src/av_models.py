import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logit_scipy import Logit

class av_base(Logit):
    def __init__(self):
        # Define the parameter names and initial values
        param_names = ['audR', 'audL', 'visR', 'visL', 'gamma', 'bias']
        param_init = {
            'bias': 0
        }

        param_bounds = {
            "gamma": (0.2, 2),
        }
        
        # Call parent class's init to handle the parameter setup
        super().__init__(param_names, param_init,param_bounds)

    def predict_log_proba(self, X):
        self.check_params()  # Ensure all params are initialized

        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        # Calculate log odds
        log_odds = (
            self.params['visL'] * (vL ** self.params['gamma']) +
            self.params['visR'] * (vR ** self.params['gamma']) +
            self.params['audL'] * aL + self.params['audR'] * aR +
            self.params['bias']
        )
        return log_odds
    
    @staticmethod
    def base_pseudo():
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

        return all_pseudo

    def generate_pseudo_trials(self):
        # this is a utility function that gets modified in sister classes # (now it seems it has not use but it does)
        self.pseudo = self.base_pseudo()

    def plot_pseudo_predictions(
        self, yscale="log", ax=None, predplotkwargs={"ls": "-"}, **kwargs
    ):

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))

        self.generate_pseudo_trials(**kwargs)

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(self.pseudo)))

        gamma = self.params['gamma']

        for a, mycolor in zip(self.pseudo, colors):
            x_ = a["visR"] ** gamma - a["visL"] ** gamma
            y_ = self.predict_proba(a)

            if yscale == "log":
                y_ = np.log(y_ / (1 - y_))

            ax.plot(x_, y_, color=mycolor, **predplotkwargs)
