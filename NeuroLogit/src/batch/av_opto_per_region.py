
# this bascially take all the subjects and fits them. things to take care of: 
# when some subjects have not been inactivated on both hemispheres
# output is 1) scores 2) parameters

# %%
import pandas as pd
from datetime import datetime

from floras_helpers.io import get_current_timestamp

from utils.av_dat_utils import get_paths, preproc_av_opto_data, filt_split_trials
import src.models.av_models_opto as model_set


def av_batch_fit(set_name = r'opto\Rinberg',reformat = False):
 
 
    if reformat: 
        preproc_av_opto_data(set_name)

    _,formatted_path,save_path = get_paths(set_name)


    region_paths = list(formatted_path.glob('*.csv'))
    models_to_fit = ['av_opto_hemispheric_additive','av_opto_hemispheric_divisive']

    results = []
    for region_path in region_paths:
        trials = pd.read_csv(region_path)
        subjects = trials.subject.unique()

        for mouse in subjects:
            trials_of_subject = trials[trials.subject == mouse]
            X_tot,X_train,X_test, y_train,y_test = filt_split_trials(trials_of_subject)

            for model_name in models_to_fit: 
                m  = getattr(model_set,model_name)
                m = m()
                m.fit(X_train,y_train)

                params = pd.DataFrame(m.params,index =[0])

                test_is_opto = (X_test.bias_opto == 1)
                params['log_loss'] = m.score(X_test,y_test,scorer = 'log_loss')
                params['auc'] = m.score(X_test,y_test,scorer = 'roc_auc_score')

                params['log_loss_opto'] = m.score(X_test[test_is_opto],y_test[test_is_opto],scorer = 'log_loss')
                params['auc_opto'] = m.score(X_test[test_is_opto],y_test[test_is_opto],scorer = 'roc_auc_score')
                params['mouseID'] = mouse
                params['model'] = model_name  
                params['region'] = region_path.stem
                results.append(params)
                



    results = pd.concat(results, ignore_index=True)


    results.to_csv(save_path / f'summary_{get_current_timestamp()}.csv')

# %%
if __name__ == "__main__":
    av_batch_fit()
