# NeuroLogit
Scikit-learn style logistic classification that was developed to probe the brain in two-alternative forced choice tasks in mice. 

Key features are: 
 - define any non-linear combination of predictors before the classifier step (e.g. a fitted contrast-sensitivity power law), by writing your own `predict_log_proba`
 - test the contribution of individual parameters with ease, by fixing them to set values
 - expanded to a 3-choice classifier to fit NoGo choices, which occur frequently in 2-AFC tasks (see [my preprint](https://www.biorxiv.org/content/10.64898/2026.06.05.730072v2))

## System requirements
- Tested on Win 11/python 3.10 
- Dependencies compiled in the `setup.py`
- No non-standard hardware required

## Installation
```
git clone https://github.com/takacsflora/NeuroLogit
cd NeuroLogit
pip install -e .
```
To use NeuroLogit from another project without cloning: `pip install git+https://github.com/takacsflora/NeuroLogit.git`


Typical install time: <2 mins

## Demos
1. To fit models to mouse choices during an audiovisual decision making task, see the tutorial [here](https://github.com/takacsflora/NeuroLogit/blob/main/NeuroLogit/tutorial/audiovisual_mice.ipynb). Includes models to assess brain inactivations on real data form mouse decisions in an audiovisual task. 

2. To build your own model for your own behavioural paradigm, see the tutorial [here](https://github.com/takacsflora/NeuroLogit/blob/main/NeuroLogit/tutorial/building_your_own_class.ipynb). 

3. The description of code for neural analysis is located [here](https://github.com/takacsflora/NeuroLogit/blob/main/NeuroLogit/tutorial/neural_data.ipynb). Neural data is available upon request at the moment. 

(typically all of these models fit in <2 mins)

## License
MIT. 