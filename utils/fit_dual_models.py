# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

dat_path = r"D:\LogRegression\opto\Rinberg\formatted\SC.csv"
trials = pd.read_csv(dat_path)

# this function can poetntially be moved to dat utils


def wrap_extraction(trials):
    stim_predictors = ["visR", "visL", "audR", "audL", "bias"]
    
    opto_predictors = [
        "visR_opto",
        "visL_opto",
        "audR_opto",
        "audL_opto",
        "bias_opto",
        "hemisphere",
    ]

    all_predictors = stim_predictors + opto_predictors

    # filter the trial matrix
    trials = trials[
        (trials.choice == 0) | (trials.choice == 1)
    ]  # keep only the post-stim correct trials
    n_trials = trials.bias_opto.value_counts().min()
    trials_ctrl = trials[trials.bias_opto == 0].sample(n_trials, random_state=1)
    trials_opto = trials[trials.bias_opto == 1].sample(n_trials, random_state=1)
    trials = pd.concat([trials_ctrl, trials_opto])

    X = trials[all_predictors]
    y = trials["choice"]
    stratifyIDs = trials.trialtype_id
    # stratifyIDs = stratifyIDs.fillna(100) # nan means control trials

    # balance opto & non-opto trials

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, shuffle=True, stratify=stratifyIDs
    )

    return trials, X_train, X_test, y_train, y_test


def plot_data(X_tot, **plotkwargs):

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    plot_psychometric(X_tot[X_tot.bias_opto == 0], ax=ax[0], **plotkwargs)
    plot_psychometric(
        X_tot[(X_tot.bias_opto == 1) & (X_tot.hemisphere == 1)], ax=ax[1], **plotkwargs
    )
    plot_psychometric(
        X_tot[(X_tot.bias_opto == 1) & (X_tot.hemisphere == -1)], ax=ax[2], **plotkwargs
    )
    plot_psychometric(
        X_tot[(X_tot.bias_opto == 1) & (X_tot.hemisphere == 0)], ax=ax[3], **plotkwargs
    )

    return fig, ax


# %%
# for subject in subjects:
# now it is extrememly slow!!! (maybe it is more about the amount of data that we suddenrly put in?)
import test_scipy
from plot_utils import plot_psychometric

trials_of_subject = trials[trials.subject == 3]
X_tot, X_train, X_test, y_train, y_test = wrap_extraction(trials_of_subject)


model_names = [
    "AV_model",
    "AV_dual_contra_additive",
    "AV_dual_contra_divisive",
    "AV_contra_divisive",
    "AV_visBias_divisive",
    "AV_split_bias_dual",
]

results = []
# fix gamma with AV_model and use that for the rest!
for model in model_names:

    model_object = getattr(test_scipy, model)
    m = model_object()

    # if model == 'AV_model':
    #     m.fit(X_train,y_train)
    # else:
    #     m.fit(X_train,y_train,fixed_params={'gamma':fixed_gamma})

    m.fit(X_train, y_train)

    coefs = m.get_params()
    params = pd.DataFrame([coefs])
    y_pred_prob = m.predict_proba(X_test)
    y_pred = m.predict(X_test)
    params["loss"] = -log_loss(y_test, y_pred_prob)
    params["auc"] = roc_auc_score(y_test, y_pred)
    params["model"] = model

    if model == "AV_model":
        fixed_gamma = float(params.gamma.values[0].round(3))

    fig, ax = plot_data(X_tot, yscale="sig", gamma=fixed_gamma)

    options = [np.nan, 1, -1, 0]
    hemispheres = ["ctrl", "right inactivated", "left inactivated", "both inactivated"]
    for i, (o, myname) in enumerate(zip(options, hemispheres)):
        if model == "AV_model":
            m.plot_pseudo_predictions(yscale="sig", ax=ax[i])

        else:
            m.plot_pseudo_predictions(yscale="sig", ax=ax[i], hemisphere=o)
        ax[i].set_title(myname)
    # fig.suptitle(model)
    results.append(params)


results = pd.concat(results, ignore_index=True)
# for each subject possibly
# %%
results

# %%
