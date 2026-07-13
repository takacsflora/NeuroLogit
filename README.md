# NeuroLogit
Scikit-learn style logistic classification that was developed to probe the brain in two-alternative forced choice tasks in mice. Key features are: 
 - define any non-linear combination of predictors before the classifier step (e.g. a fitted contrast-sensitivity power law), by writing your own `predict_log_proba`
 - test the contribution of individual parameters with ease, by fixing them to set values
 - expanded to a 3-choice classifier to fit NoGo choices, which occur frequently in 2-AFC tasks (see [my preprint](https://www.biorxiv.org/content/10.64898/2026.06.05.730072v1))

If you want to fit models to mouse choices during an audiovisual decision making task, see the tutorial here. Includes a tutorial to assess brain inactivations. 

If you want to build your own model for your own behavioural paradigm, see the tutorial here. 

If you want to see how I tried assessing the the contribution of neural activity to choice, see here. 
