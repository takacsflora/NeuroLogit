
from NeuroLogit.src.ephys.encoding_avg.batch_proc import get_predictors,get_tested_models

import numpy as np 
import pandas as pd

def get_component_matrix(**kwargs):

    model_names = get_tested_models(**kwargs)

    # get unique predictors from all models

    components = [] # a pre_determined list of components that will keep a sequence

    # Get all unique components across all models
    all_components = set()
    for model in model_names:
        all_components.update(get_predictors(model))

    all_components = sorted(all_components)

    # # Build a boolean matrix: rows=components, cols=models
    component_matrix = np.zeros((len(all_components), len(model_names)), dtype=bool)
    for col, model in enumerate(model_names):
        predictors = set(get_predictors(model))
        for row, comp in enumerate(all_components):
            if comp in predictors:
                component_matrix[row, col] = True

    # Convert to DataFrame for easier handling
    component_df = pd.DataFrame(component_matrix, index=all_components, columns=model_names)

    return component_df
