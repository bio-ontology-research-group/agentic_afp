import pandas as pd

mlp = pd.read_pickle('data/test_predictions_mlp.pkl')
dm = pd.read_pickle('data/test_predictions_diamond.pkl')

# combine scores
for ont in ['mf', 'bp', 'cc']:
    mlp_preds = mlp[f"{ont}_preds"].values
    dm_preds = dm[f"{ont}_preds"].values
    combined_preds = []
    for mlp_s, dm_s in zip(mlp_preds, dm_preds):
        combined_preds.append(0.5 * (mlp_s + dm_s))
    mlp[f"{ont}_preds"] = combined_preds

mlp.to_pickle('data/test_predictions_combined.pkl')