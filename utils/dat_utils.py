

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

def get_paths(set_name):
    """
    Creates standard path structure for data_management and can recall Paths
    Parameters: 
        set_name: str
            identifier of the dataset that points to the raw data 

    Returns:
        basepath,formatted_path,savepath 
        pathlib.Paths 
        raw data, processed data for fitting, results of fitting
    """
    basepath = Path(f"D:\LogRegression\{set_name}")
    formatted_path = basepath / 'formatted'
    formatted_path.mkdir(parents=False,exist_ok=True)
    savepath = formatted_path / 'fit_results'
    savepath.mkdir(parents=False,exist_ok=True)

    return basepath,formatted_path,savepath

def preproc_av_opto_data(trials):
    # todo# preproc it even further by reading directly the data...
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
    


def get_benchmark_opto_dataset(subject=1):
    """
    allows the easy call of an example dataset, i.e. subject 1
    """
    dat_path = r"D:\LogRegression\opto\Rinberg\formatted\SC.csv"
    trials = pd.read_csv(dat_path)
    trials_of_subject = trials[trials.subject == subject]

    return preproc_av_opto_data(trials_of_subject)