# for computations
import numpy as np
import pandas as pd

# for plot handling
import matplotlib.pyplot as plt

# because taking the log inevitably results in runtime errors
np.seterr(divide="ignore")


def plot_psychometric(
    trials, gamma=1, yscale="log", ax=None, dataplotkwargs={"marker": "o", "ls": ""}
):
    """
    plot the model prediction for this specific model
    if the model has neural components we 0 those out

    refit: bool
        whether to refit the AV model -- in this case we use the refitted model weights as predictions

    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    assert np.isin(
        ["visR", "audR", "visL", "audL", "choice"], trials.columns
    ).all(), "the trials are inputted in an unexpected format."
    # handling gamma -- for now we use common gamma
    trials = trials.copy()
    for v in ["visR", "visL"]:
        trials.loc[:, f"{v}_gamma"] = trials[v].values ** gamma

    # get commonly used variables
    visDiff = trials.visR_gamma - trials.visL_gamma
    audDiff = trials.audR - trials.audL
    choices = trials.choice
    Vs = np.unique(visDiff)
    As = np.unique(audDiff)

    # determine the colors we will use in the plot using coolwarm as colormap
    colors = plt.cm.coolwarm(np.linspace(0, 1, As.size))

    Vmesh, Amesh = np.meshgrid(Vs, As)

    for v, a, mycolor in zip(Vmesh, Amesh, colors):
        x = v
        y = np.array(
            [
                np.mean(choices[(visDiff == vi) & (audDiff == ai)])
                for vi, ai in zip(v, a)
            ]
        )
        if yscale == "log":
            y = np.log(y / (1 - y + 1e-10))

        ax.plot(x, y, color=mycolor, **dataplotkwargs)

    # plot the guider lines
    if yscale == "log":
        ax.axhline(0, color="k", ls="--")
        ax.set_ylim([-10, 10])
    else:
        ax.axhline(0.5, color="k", ls="--")
        ax.set_ylim([-0.1, 1.1])

    ax.axvline(0, color="k", ls="--")
