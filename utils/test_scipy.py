# %%

# collection on LogisticClassification models that rely on scipy.optimise.minimise
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt


class AV_model:
    def __init__(self):
        # Initialize sensory parameters
        self.audR = 1
        self.audL = 1
        self.visR = 1
        self.visL = 1

        # initialise other paarameters
        self.gamma = 1 
        self.bias = 0

        # self.param_list = 

    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values

        log_odds = (
            self.visL * (vL ** self.gamma) +
            self.visR * (vR ** self.gamma) +
            self.audL * aL + self.audR * aR +
            self.bias
        )
        return log_odds
    

    @staticmethod
    def raise_to_sigm(logOdds):
        "function to calculate pR from logOdds"
        return np.exp(logOdds) / (1 + np.exp(logOdds))

    def predict_proba(self, X):
        return self.raise_to_sigm(self.predict_log_proba(X))

    def predict(self, X):
        return (self.predict_log_proba(X) > 0).astype('float')

    def get_params(self):
        added_params = ['audR', 'audL', 'visR', 'visL', 'gamma', 'bias']
        return {k: float(getattr(self, k)) for k in added_params}

    def set_params(self, params):
        # Ensure all values are standard floats when setting parameters
        for key, value in params.items():
            setattr(self, key, float(value)) 


    def get_bounds(self):
        # common dictionary to set bounds
        return {
            "audR": (None, None),
            "audL": (None, None),
            "visR": (None, None),
            "visL": (None, None),
            "gamma": (0.2, 2),
            "bias": (None, None),
        }



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

        # Set the full parameters in the model
        self.set_params(full_params)

        # Get predictions and calculate loss
        predictions = self.predict_proba(X) 
        return log_loss(y, predictions, normalize=True)


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

        # Convert to lists for minimize
        trainable_param_starting_points = list(param_mask.values())
        trainable_param_names = list(param_mask.keys())

        # Define bounds (if needed)
        bound_dict = self.get_bounds()
        trainable_param_bounds = [bound_dict[key] for key in trainable_param_names]
        # Run optimization

        result = minimize(
            self._objective,
            trainable_param_starting_points,
            args=(X, y, param_mask, fixed_params),
            bounds=trainable_param_bounds,
        )

        # Set the optimized parameters in the model
        optimized_params = {k: v for k, v in zip(trainable_param_names, result.x)}
        
        self.set_params(optimized_params)

        return result

    def score(self, X, y):
        """
        Score the model (e.g., accuracy or log-loss).
        """
        predictions = self.predict_proba(X)
        accuracy = np.mean((predictions > 0.5) == y)
        return accuracy

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

        for a, mycolor in zip(self.pseudo, colors):
            x_ = a["visR"] ** self.gamma - a["visL"] ** self.gamma
            y_ = self.predict_proba(a)

            if yscale == "log":
                y_ = np.log(y_ / (1 - y_))

            ax.plot(x_, y_, color=mycolor, **predplotkwargs)


class AV_opto_model(AV_model):
    def __init__(self):
        # Call the base model's constructor
        super().__init__()

        # Initialize opto-related parameters
        self.audR_opto = 0
        self.audL_opto = 0
        self.visR_opto = 0
        self.visL_opto = 0
        self.bias_opto = 0

    # Override the get_params method to include opto parameters
    def get_params(self):
        params = super().get_params()
        added_params = ['audR_opto', 'audL_opto', 'visR_opto', 'visL_opto', 'bias_opto']

        added_params = {k: float(getattr(self, k)) 
                       for k in added_params}

        params.update(added_params)

        return params

    def get_bounds(self):
        bounds = super().get_bounds()
        bounds.update(
            {
                "audR_opto": (None, None),
                "audL_opto": (None, None),
                "visR_opto": (None, None),
                "visL_opto": (None, None),
                "bias_opto": (None, None),
            }
        )
        return bounds

    # Override the predict_log_proba method to account for opto functionality
    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        O = X[["bias_opto"]].values  # opto modulation factor

        # Power visual parameters to gamma
        vL_ = vL**self.gamma
        vR_ = vR**self.gamma

        # Compute signal without opto
        S = (
            self.visL * vL_
            + self.visR * vR_
            + self.audL * aL
            + self.audR * aR
            + self.bias
        )

        # Compute signal with opto modulation
        S_opto = (
            self.visL_opto * vL_ * O
            + self.visR_opto * vR_ * O
            + self.audL_opto * aL * O
            + self.audR_opto * aR * O
            + self.bias_opto
        )

        # Calculate logits for probabilities
        logOdds = S + S_opto

        return logOdds

    def generate_pseudo_trials(bias_opto=0):
        pass


class AV_dual_contra_divisive(AV_model):
    def __init__(self):
        # Call the base model's constructor
        super().__init__()

        self.optoL = 0
        self.optoR = 0

    def get_params(self):
        params = super().get_params()
        params.update(
            {
                "optoL": float(self.optoL),
                "optoR": float(self.optoR),
            }
        )
        return params

    def get_bounds(self):
        bounds = super().get_bounds()
        bounds.update(
            {
                "optoL": (0, None),
                "optoR": (0, None),
            }
        )
        return bounds

    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")
        oL = (X["hemisphere"].values <= 0).astype("float")

        # Power visual parameters to gamma
        vL_ = vL**self.gamma
        vR_ = vR**self.gamma

        # Compute signal without opto
        left_odds = (self.visL * vL_ + self.audL * aL + self.bias) / (
            1 + self.optoL * oR
        )[:, np.newaxis]
        right_odds = (self.visR * vR_ + self.audR * aR + self.bias) / (
            1 + self.optoR * oL
        )[:, np.newaxis]

        # Calculate logits for probabilities
        logOdds = left_odds + right_odds

        return logOdds

    def generate_pseudo_trials(self, hemisphere=1):
        pseudo = self.base_pseudo()

        new = []
        for a in pseudo:
            a["hemisphere"] = hemisphere
            new.append(a)

        self.pseudo = new


class AV_visBias_divisive(AV_dual_contra_divisive):
    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")
        oL = (X["hemisphere"].values <= 0).astype("float")

        # Power visual parameters to gamma
        vL_ = vL**self.gamma
        vR_ = vR**self.gamma

        # Compute signal without opto
        left_odds = (
            (self.visL * vL_ + self.bias) / (1 + self.optoL * oR)[:, np.newaxis]
        ) + self.audL * aL
        right_odds = (
            (self.visR * vR_ + self.bias) / (1 + self.optoR * oL)[:, np.newaxis]
        ) + self.audR * aR

        # Calculate logits for probabilities
        logOdds = left_odds + right_odds

        return logOdds


class AV_contra_divisive(AV_model):
    def __init__(self):
        # Call the base model's constructor
        super().__init__()

        self.optoL = 0
        self.biasR = 0

    def get_params(self):
        params = super().get_params()
        params.update(
            {
                "optoL": float(self.optoL),
                "biasR": float(self.optoL),
            }
        )
        return params

    def get_bounds(self):
        bounds = super().get_bounds()
        bounds.update(
            {
                "optoL": (None, None),
                "biasR": (None, None),
            }
        )
        return bounds

    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")
        oL = (X["hemisphere"].values <= 0).astype("float")

        # Power visual parameters to gamma
        vL_ = vL**self.gamma
        vR_ = vR**self.gamma

        # Compute signal without opto
        left_odds = (self.visL * vL_ + self.audL * aL + self.bias) / (
            1 + self.optoL * oR
        )[:, np.newaxis]
        right_odds = (self.visR * vR_ + self.audR * aR + self.biasR) / (
            1 + self.optoL * oL
        )[:, np.newaxis]

        # Calculate logits for probabilities
        logOdds = left_odds + right_odds

        return logOdds

    def generate_pseudo_trials(self, hemisphere=1):
        pseudo = self.base_pseudo()

        new = []
        for a in pseudo:
            a["hemisphere"] = hemisphere
            new.append(a)

        self.pseudo = new


class AV_dual_contra_additive(AV_model):
    def __init__(self):
        # Call the base model's constructor
        super().__init__()

        # Initialize opto-related parameters
        self.audR_opto = 0
        self.audL_opto = 0
        self.visR_opto = 0
        self.visL_opto = 0
        self.optoL = 0
        self.optoR = 0

    def get_params(self):
        params = super().get_params()
        params.update(
            {
                "audR_opto": float(self.audR_opto),
                "audL_opto": float(self.audL_opto),
                "visR_opto": float(self.visR_opto),
                "visL_opto": float(self.visL_opto),
                "optoL": float(self.optoL),
                "optoR": float(self.optoR),
            }
        )
        return params

    def get_bounds(self):
        bounds = super().get_bounds()
        bounds.update(
            {
                "audR_opto": (None, None),
                "audL_opto": (None, None),
                "visR_opto": (None, None),
                "visL_opto": (None, None),
                "optoL": (None, None),
                "optoR": (None, None),
            }
        )
        return bounds

    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")[:, np.newaxis]
        oL = (X["hemisphere"].values <= 0).astype("float")[:, np.newaxis]

        # Power visual parameters to gamma
        vL_ = vL**self.gamma
        vR_ = vR**self.gamma

        # Compute signal without opto
        left_odds = self.visL * vL_ + self.audL * aL
        left_opto_odds = self.visL_opto * vL_ * oR + self.audL_opto * aL * oR
        right_odds = self.visR * vR_ + self.audR * aR
        right_opto_odds = self.visR_opto * vR_ * oL + self.audR_opto * aR * oL
        bias_odds = self.bias + self.optoL * oR + self.optoR * oL

        # Calculate logits for probabilities
        logOdds = left_odds + right_odds + left_opto_odds + right_opto_odds + bias_odds

        return logOdds

    def generate_pseudo_trials(self, hemisphere=1):
        pseudo = self.base_pseudo()

        new = []
        for a in pseudo:
            a["hemisphere"] = hemisphere
            new.append(a)

        self.pseudo = new


class AV_split_bias_dual(AV_dual_contra_divisive):
    def __init__(self):
        # Call the base model's constructor
        super().__init__()
        self.biasR = 0

    def get_params(self):
        params = super().get_params()
        params.update(
            {
                "biasR": float(self.optoL),
            }
        )
        return params

    def get_bounds(self):
        bounds = super().get_bounds()
        bounds.update(
            {
                "biasR": (None, None),
            }
        )
        return bounds

    def predict_log_proba(self, X):
        # Extract inputs
        vL = X[["visL"]].values
        vR = X[["visR"]].values
        aL = X[["audL"]].values
        aR = X[["audR"]].values
        oR = (X["hemisphere"].values >= 0).astype("float")
        oL = (X["hemisphere"].values <= 0).astype("float")

        # Power visual parameters to gamma
        vL_ = vL**self.gamma
        vR_ = vR**self.gamma

        # Compute signal without opto
        left_odds = (self.visL * vL_ + self.audL * aL + self.bias) / (
            1 + self.optoL * oR
        )[:, np.newaxis]
        right_odds = (self.visR * vR_ + self.audR * aR + self.biasR) / (
            1 + self.optoR * oL
        )[:, np.newaxis]

        # Calculate logits for probabilities
        logOdds = left_odds + right_odds

        return logOdds
