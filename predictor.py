from preprocessor import preprocessor
import pandas as pd
import pickle
from scipy.interpolate import interp1d
import numpy as np

def prediction(input_path, output_path):
    df= pd.read_csv(input_path)
    df = preprocessor(df)

    #The ratios we decided to use
    variables_used = ["current_ratio",
                    "Adj_NetIncome_to_Assets",
                    "asset_turnover_ratio",
                    "tie",     
                    "cash_and_equiv",
                    "eqty_tot",               
                        "Liabilities_TangibleAssets_Ratio",
                    "ocf_ratio" ]

    df = df[variables_used]

    # Load LOWESS metadata
    with open("meta_data_for_lowess_final.pkl", "rb") as f:
        loaded_metadata = pickle.load(f)

    # Apply LOWESS PD transformation using precomputed metadata.
    def predict_lowess_pd_efficient(raw_values, metadata):

        raw_values = np.asarray(raw_values, dtype=float)

        # Metadata
        bin_mins = np.array(metadata["bin_mins"])
        bin_maxs = np.array(metadata["bin_maxs"])
        xs = np.array(metadata["xs"])
        ys = np.array(metadata["ys"])
        k = metadata["k"]

        # LOWESS function reconstruction
        smooth_func = interp1d(xs, ys, fill_value="extrapolate", bounds_error=False)

        # Assign bins (equal-sized bins via bin_maxs)
        bin_idx = np.searchsorted(bin_maxs, raw_values, side="right")
        bin_idx = np.clip(bin_idx, 0, k - 1)

        # Bin-level min/max for each sample
        bin_min = bin_mins[bin_idx]
        bin_max = bin_maxs[bin_idx]

        # Continuous scaled position inside bin
        with np.errstate(divide="ignore", invalid="ignore"):
            x_scaled = (
                bin_idx / k
                + (raw_values - bin_min) / (bin_max - bin_min) * (1 / k)
            )

            # Handle zero-width bins (all values identical in that bin)
            x_scaled = np.where(bin_max > bin_min, x_scaled, bin_idx / k)

        # Predict PD via interpolated LOWESS curve
        pd_vals = smooth_func(x_scaled)

        return pd_vals
    

    #Reconstruct and apply LOWESS transformations for all variables
    def apply_lowess_all(df, metadata_dict):

        df_out = df.copy()

        for col, meta in metadata_dict.items():
            # Apply your lowess function
            raw_vals = df_out[col].values
            transformed_vals = predict_lowess_pd_efficient(raw_vals, meta)

            # Name for new column
            new_col = f"transformed_{col}"
            df_out[new_col] = transformed_vals
            df_out.drop(columns=[col], inplace=True)

        return df_out
    
    df = apply_lowess_all(df, loaded_metadata)

    with open("final_gam_model.pkl", "rb") as f:
        final_gam_model = pickle.load(f)

    # Prediction function
    def estimate(df, model):
        return model.predict_proba(df)

    pred = estimate(df, final_gam_model)

    with open("calibration_metadata.pkl", "rb") as f:
        calibration_metadata = pickle.load(f)

    #First step of calibration
    calibrated_pd = predict_lowess_pd_efficient(pred, calibration_metadata["pred_test"])

    #Second step of calibration
    def calibrate_2(pred):
        T = 0.028    # true base rate based on https://www.bancaditalia.it/pubblicazioni/relazione-annuale/2012/en_rel_2012.pdf?language_id=1&utm
        S = 0.013575787769336156   # sample base rate
        
        numerator = T * (pred - pred * S)
        denominator = (S 
                    - pred * S 
                    + pred * T 
                    - S * T)
        
        calibrated = numerator / denominator
        return calibrated

    final_pred = calibrate_2(calibrated_pd)

    pd.DataFrame(final_pred).to_csv(output_path, index=False, header=False)



