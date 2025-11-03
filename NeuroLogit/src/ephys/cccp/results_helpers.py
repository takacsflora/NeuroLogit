
#%%
# constructing helper function to get ipsi vs contra
def add_ipsi_contra(df):
    df_cols = list(df.columns)
    df['hemi_name'] = df['hemi'].map({-1: 'Right', 1: 'Left'})
    # swap firing rates 
    right_columns = [col for col in df.columns if 'Right' in col]
    for right_col in right_columns:
        left_col = right_col.replace('Right', 'Left')
        contra_name = right_col.replace('Right', 'contra')
        ipsi_name = right_col.replace('Right', 'ipsi')
        df[contra_name] = df.apply(lambda row: row[left_col] if row['hemi_name'] == 'Right' else row[right_col], axis=1)
        df[ipsi_name] = df.apply(lambda row: row[right_col] if row['hemi_name'] == 'Right' else row[left_col], axis=1)

    # swap the cp/AUC values
    cp_columns = [col for col in df.columns if 'cp' in col]
    for cp_col in cp_columns:
        df[cp_col] = df.apply(lambda row: row[cp_col] if row['hemi_name'] == 'Left' else 1 - row[cp_col], axis=1)

    # recalculate the angles and the magnitudes of the response vectors
    return df

# %%
