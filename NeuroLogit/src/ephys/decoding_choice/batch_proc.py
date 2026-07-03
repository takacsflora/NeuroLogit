# here the idea is that gamma is very hard to fit even session to session
# so we fit one gamma per mice, then we obtain the sensory parameters on a session by session basis
# this is done in src.behav.av_logit_per_session_ephys.py

# now we just use these as stimOdds to fit the neural data
from pathlib import Path
from src.ephys.dat_utils import get_ephys_dataset

from NeuroLogit.src.ephys.decoding_choice.fit_models_oldsklearn import fit_session_sklearn
from NeuroLogit.src.ephys.decoding_choice.fit_models import fit_session_scipy

def get_time_kws(which):
    if which == 'prestim':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.2,'post_time':0.0}}
    elif which == 'stim':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.0,'post_time':0.2}}
    elif which == 'choice':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.2,'post_time':0.0}}    
    elif which == 'prestim0':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0,'post_time':0.05}}
    elif which == 'prestim1':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.05,'post_time':0.0}}
    elif which == 'prestim2':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.1,'post_time':-0.05}}
    elif which == 'prestim3':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.15,'post_time':-0.1}}
    elif which == 'prestim4':
        return {'load_raster':'prestim','avg_kwargs':{'pre_time':0.2,'post_time':-0.15}}
    elif which == 'choice0':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':-0.05,'post_time':0.1}}
    elif which == 'choice1':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.0,'post_time':0.05}}
    elif which == 'choice2':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.05,'post_time':0.0}}
    elif which == 'choice3':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.1,'post_time':-0.05}}
    elif which == 'choice4':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.15,'post_time':-0.1}}
    elif which == 'choice5':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.2,'post_time':-0.15}}
    elif which == 'choice6':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.25,'post_time':-0.2}}    
    elif which == 'choice7':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.3,'post_time':-0.25}}
    elif which == 'choice8':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.35,'post_time':-0.3}}
    elif which == 'choice9':
        return {'load_raster':'choice','avg_kwargs':{'pre_time':0.4,'post_time':-0.35}}
    else:
        raise ValueError("Unknown time window")

def batch_fit_sessions(recompute=False,model_type='multinomial'):
    """fit all sessions and save the results"""

    save_path = Path(r'D:\AV_Neural_Data_Sept2025\decoding_choice_results')
    sessions = get_ephys_dataset(set_name='all', subset='')
    for i,rec in sessions.iterrows():
        session_path = save_path / f"{rec.subject}_{rec.date}"

        if (not session_path.exists()) or recompute:
            try: 
                print(f"fitting {rec.subject} {rec.date}")

                session_path.mkdir(exist_ok=True,parents=True)
                times = ['prestim','choice']
                times += [f'choice{i}' for i in range(9)] # finer bins around choice
                times += [f'prestim{i}' for i in range(4)] # finer bins around prestim
                for t in times: 
                    time_kws = get_time_kws(t)
                    
                    if 'scipy' in model_type:
                        encoder_based_neur_selecion = model_type.split('_')[1] if '_' in model_type else 'choice'
                        feature_based_selection = model_type.split('_')[2] if '_' in model_type and len(model_type.split('_'))>2 else 'none'

                        task_ev, weights, metrics = fit_session_scipy(rec.subject,rec.date,time_kws,VE_choice_thr=0.005,encoder_model_type='passive_active_Ridge10',
                                                                      encoder_based_seletion =encoder_based_neur_selecion,
                                                                      feature_based_selection=feature_based_selection)
                        
                    else:
                        task_ev, weights, metrics = fit_session_sklearn(rec.subject,rec.date,multi_class=model_type,**time_kws)
                    
                    
                    # no need to save all the neuron columns as they will vary across sessions
                    keep_cols = [c for c in task_ev.columns if 'neuron_' not in c]
                    task_ev = task_ev[keep_cols].copy()

                    # add subject, date and session ID to the df
                    task_ev['subject'] = rec.subject
                    task_ev['date'] = rec.date
                    task_ev['sessionID'] = f"{rec.subject}_{rec.date}"

                    weights['subject'] = rec.subject
                    weights['date'] = rec.date
                    weights['sessionID'] = f"{rec.subject}_{rec.date}"

                    metrics['subject'] = rec.subject
                    metrics['date'] = rec.date
                    metrics['sessionID'] = f"{rec.subject}_{rec.date}"
                    
                    # save the results
                    timing_stub = f'{t}_pre{time_kws["avg_kwargs"]["pre_time"]}_post{time_kws["avg_kwargs"]["post_time"]}'
                    task_ev.to_csv(session_path / f'task_ev_{model_type}_{timing_stub}.csv',index=True)
                    weights.to_csv(session_path / f'weights_{model_type}_{timing_stub}.csv',index=True)
                    metrics.to_csv(session_path / f'metrics_{model_type}_{timing_stub}.csv',index=False)
            except Exception as e:
                print(f'error processing {rec.subject} {rec.date}: {e}')
                with open(session_path / 'error_log.txt', 'w') as f:
                    f.write(str(e))

if __name__ == "__main__":
    batch_fit_sessions(recompute=True, model_type='scipy_all_l1')


