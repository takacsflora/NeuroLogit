from pathlib import Path
import pandas as pd
import time

from sklearn.metrics import roc_auc_score,log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from test_scipy import AV_opto_model
from test_sklearn import fit_model,get_weights
from test_torch import fit_pytorch_model

def fit_opto_model(rec,nametag=None,gammafit=False,scipyfit = False):
    t0 = time.time()

    if isinstance(rec, (str, Path)):
        print('fitting', rec, '...')
        trials = pd.read_csv(rec)  # Load trials from the CSV file
        subject = rec.name.split('_')[0]

    elif isinstance(rec, pd.DataFrame):
        trials = rec  # Use the DataFrame directly 
        if nametag: subject=nametag
        else: subject = 'test'


    stim_predictors = ['visR','visL','audR','audL','bias']
    opto_predictors = ['visR_opto','visL_opto','audR_opto','audL_opto','bias_opto']    

    all_predictors =  stim_predictors + opto_predictors

    #filter the trial matrix
    trials = trials[(trials.choice==0) | (trials.choice==1)] # keep only the post-stim correct trials

    X = trials[all_predictors]
    y = trials['choice']
    stratifyIDs = trials['trialtype_id']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1,shuffle=True,stratify=stratifyIDs)
        
    if gammafit:
        # in this case the sklearn package fits a gamma value
        m = fit_model(X,y,gridCV_vis=True)
        w = get_weights(m)
        params = w['weights']
        params['gamma'] = float(w['hyperparameters']['feature_selector__vis__power'])
        gamma_fitted = True

    
    if scipyfit:
        m = AV_opto_model()        
        if gamma_fitted:
            fixed_params={'gamma':params['gamma']}
            m.fit(X_train,y_train,fixed_params=fixed_params)
        else:
            m.fit(X_train,y_train)
        coefs = m.get_params()
        params = pd.DataFrame([coefs])

    if (not gammafit) & (not scipyfit):
        m = LogisticRegression(fit_intercept=False)
        m.fit(X_train,y_train)
        params = pd.DataFrame(m.coef_,columns=all_predictors)

    passed_time = time.time()-t0
    print(f'fitted in {passed_time} secs')


    y_pred_prob = m.predict_proba(X_test)
    y_pred = m.predict(X_test)
    neg_log_loss = -log_loss(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test,y_pred)

    return params,neg_log_loss,auc_score,passed_time