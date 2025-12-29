import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from pygam import LogisticGAM, s
from functools import reduce
import operator
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import roc_curve, roc_auc_score
from preprocessor import preprocessor
from tqdm import tqdm
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def estimator(df):

    def create_def_flag(df):
        #Function to create def_flag based on stmt_date and def_date
        def create_def_flag(df):

            df['stmt_date'] = pd.to_datetime(df['stmt_date'], errors='coerce') 
            df['def_date'] = pd.to_datetime(df['def_date'], errors='coerce')
            df = df[df['stmt_date'] != '2012-12-31']

            # Container for all results
            final_list = []

            # Loop over each ID with tqdm progress bar
            for id_val, group_df in tqdm(df.groupby('id'), total=df['id'].nunique(), desc="Processing IDs"):
                group_df = group_df.copy()  # avoid SettingWithCopyWarning

                # Case 1: all def_date missing
                if group_df['def_date'].isna().all():
                    group_df['def_flag'] = 0
                    final_list.append(group_df)
                
                else:
                    # Get the unique def_date for this ID
                    def_dt = group_df['def_date'].dropna().iloc[0]

                    # Case 2: def_date in June-Dec
                    if def_dt.month >= 6:
                        prev_year_stmt = pd.Timestamp(year=def_dt.year - 1, month=12, day=31)

                    # Case 3: def_date in Jan-May
                    else:
                        prev_year_stmt = pd.Timestamp(year=def_dt.year - 2, month=12, day=31) 

                    # Keep rows with stmt_date <= prev_year_stmt
                    temp_df = group_df.loc[group_df['stmt_date'] <= prev_year_stmt].copy()

                    # Set flags
                    temp_df['def_flag'] = 0
                    temp_df.loc[temp_df['stmt_date'] == prev_year_stmt, 'def_flag'] = 1
                    
                    final_list.append(temp_df)
            # Concatenate all at once at the end
            final = pd.concat(final_list, ignore_index=True)
            return final

    df = create_def_flag(df)
    df = preprocessor(df)

    # List of variables used in the model
    variables_used = ["current_ratio",
                    "Adj_NetIncome_to_Assets",
                    "asset_turnover_ratio",
                    "tie",     
                    "cash_and_equiv",
                    "eqty_tot",               
                        "Liabilities_TangibleAssets_Ratio",
                    "ocf_ratio" ]

    # LOWESS transformation with equal-number-of-firms bins.
    def lowess_transform_variable(df, ratio_col, default_col="def_flag", k=50, frac=0.25):
        temp = df[[ratio_col, default_col]].copy()
        N = len(temp)

        # 2. Sort by ratio value
        temp = temp.sort_values(by=ratio_col).reset_index(drop=True)

        # 3. Equal-sized bins
        bin_edges_idx = np.linspace(0, N, k+1, dtype=int)
        bin_idx = np.zeros(N, dtype=int)
        for i in range(k):
            start, end = bin_edges_idx[i], bin_edges_idx[i+1]
            bin_idx[start:end] = i
        temp["bin"] = bin_idx

        # Bin min/max
        bin_mins = temp.groupby("bin")[ratio_col].min().values
        bin_maxs = temp.groupby("bin")[ratio_col].max().values

        # 4. PD per bin
        qdr = temp.groupby("bin")[default_col].mean().reset_index()
        qdr["bin_scaled"] = qdr["bin"] / k

        # 5. LOWESS smoothing
        smoothed = lowess(
            endog=qdr[default_col],
            exog=qdr["bin_scaled"],
            frac=frac,
            return_sorted=True
        )
        xs, ys = smoothed[:, 0], smoothed[:, 1]

        # Smooth function
        smooth_func = interp1d(xs, ys, fill_value="extrapolate", bounds_error=False)

        # 6. Transform the FULL column
        raw_values = df[ratio_col].values

        # assign bins for full column
        bin_idx_full = np.searchsorted(bin_maxs, raw_values, side="right")
        bin_idx_full = np.clip(bin_idx_full, 0, k - 1)

        bin_min_full = bin_mins[bin_idx_full]
        bin_max_full = bin_maxs[bin_idx_full]

        # scaled position inside each bin
        with np.errstate(divide='ignore', invalid='ignore'):
            x_scaled = (
                bin_idx_full / k +
                (raw_values - bin_min_full) / (bin_max_full - bin_min_full) * (1 / k)
            )
            # handle bins with identical min/max
            x_scaled = np.where(bin_max_full > bin_min_full,
                                x_scaled,
                                bin_idx_full / k)

        # LOWESS prediction
        transformed = smooth_func(x_scaled)

        # add transformed column to df
        df[f"transformed_{ratio_col}"] = transformed

        # 7. metadata for out-of-sample use
        metadata = {
            "ratio_col": ratio_col,
            "bin_mins": bin_mins.tolist(),
            "bin_maxs": bin_maxs.tolist(),
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "k": k,
            "frac": frac,
        }
        return df, smooth_func, metadata
    
    
    #Lowess Transformations
    meta_data_all = {}
    df_lowess = df.copy()
    df_lowess, current_ratio_smooth_func, current_ratio_metadata = lowess_transform_variable(
        df_lowess, 
        'current_ratio', 
        default_col="def_flag", 
        k=50, 
        frac=100000/len(df)
    )
    meta_data_all['current_ratio'] = current_ratio_metadata

    df_lowess, adj_netincome_to_assets_smooth_func, adj_netincome_to_assets_metadata = lowess_transform_variable(
        df_lowess, 
        'Adj_NetIncome_to_Assets', 
        default_col="def_flag", 
        k=50, 
        frac=110000/len(df)
    )
    meta_data_all['Adj_NetIncome_to_Assets'] = adj_netincome_to_assets_metadata

    df_lowess, asset_turnover_ratio_smooth_func, asset_turnover_ratio_metadata = lowess_transform_variable(
        df_lowess, 
        'asset_turnover_ratio', 
        default_col="def_flag", 
        k=50, 
        frac=170000/len(df)
    )
    meta_data_all['asset_turnover_ratio'] = asset_turnover_ratio_metadata

    df_lowess, tie_smooth_func, tie_metadata = lowess_transform_variable(
        df_lowess, 
        'tie', 
        default_col="def_flag", 
        k=50, 
        frac=83000/len(df)
    )
    meta_data_all['tie'] = tie_metadata

    df_lowess, cash_and_equiv_smooth_func, cash_and_equiv_metadata = lowess_transform_variable(
        df_lowess, 
        'cash_and_equiv', 
        default_col="def_flag", 
        k=50, 
        frac=160000/len(df)
    )
    meta_data_all['cash_and_equiv'] = cash_and_equiv_metadata

    df_lowess, eqty_tot_smooth_func, eqty_tot_metadata = lowess_transform_variable(
        df_lowess, 
        'eqty_tot', 
        default_col="def_flag", 
        k=50, 
        frac=100000/len(df)
    )
    meta_data_all['eqty_tot'] = eqty_tot_metadata

    df_lowess, liabilities_tangibleassets_ratio_smooth_func, liabilities_tangibleassets_ratio_metadata = lowess_transform_variable(
        df_lowess, 
        'Liabilities_TangibleAssets_Ratio', 
        default_col="def_flag", 
        k=50, 
        frac=120000/len(df)
    )
    meta_data_all['Liabilities_TangibleAssets_Ratio'] = liabilities_tangibleassets_ratio_metadata

    df_lowess, ocf_ratio_smooth_func, ocf_ratio_metadata = lowess_transform_variable(
        df_lowess, 
        'ocf_ratio', 
        default_col="def_flag", 
        k=50, 
        frac=110000/len(df)
    )
    meta_data_all['ocf_ratio'] = ocf_ratio_metadata


    #Split train and test
    selected_cols = [f'transformed_{var}' for var in variables_used]
    df_train = df_lowess[pd.to_datetime(df_lowess['stmt_date'])<='2010-12-31']
    df_test = df_lowess[pd.to_datetime(df_lowess['stmt_date'])>'2010-12-31']

    # Build training matrix
    train_df = df_train[selected_cols + ['def_flag']].copy()
    X = train_df.drop('def_flag', axis=1).values
    y = train_df['def_flag'].values

    n_feats = X.shape[1]

    terms = reduce(operator.add, [
        s(i, n_splines=20, lam=0.1) 
        for i in range(n_feats)
    ])

    # Train GAM model
    full_gam_model = LogisticGAM(terms)
    full_gam_model.fit(X, y)
    print(" GAM trained successfully on the training dataset")

    #saves the model
    # with open("final_gam_model.pkl", "wb") as f:
    #     pickle.dump(full_gam_model, f)

    # print(" Saved: final_gam_model.pkl")

    X_test = df_test[selected_cols].values
    y_test = df_test['def_flag'].values

    pred_test = full_gam_model.predict_proba(X_test)   # returns Probability(default=1)

    # Compute ROC + AUC
    fpr, tpr, thresholds = roc_curve(y_test, pred_test)
    auc_value = roc_auc_score(y_test, pred_test)

    print(f"AUC on Test Set: {auc_value:.4f}")

    #  Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'GAM ROC (AUC = {auc_value:.4f})')

    # 45-degree line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — GAM Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    df_cal, calibration_smooth_func, calibration_metadata = lowess_transform_variable(df_test, 'pred_test', default_col="def_flag", k=50, frac=120000/len(df_lowess))
    cal_metadata={}
    cal_metadata['pred_test'] = calibration_metadata

    return meta_data_all, cal_metadata

def main():
    parser = argparse.ArgumentParser(description="Run the estimator to generate preprocessing and model files.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input training dataset CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated .pkl files.")
    args = parser.parse_args()

    # Load the input CSV file
    logging.info(f"Loading input CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # Run the estimator
    logging.info("Running the estimator...")
    meta_data_all, cal_metadata = estimator(df)

    # Save the generated .pkl files
    os.makedirs(args.output_dir, exist_ok=True)

    meta_data_path = os.path.join(args.output_dir, "meta_data_all.pkl")
    with open(meta_data_path, "wb") as f:
        pickle.dump(meta_data_all, f)
    logging.info(f"Saved: {meta_data_path}")

    cal_metadata_path = os.path.join(args.output_dir, "calibration_metadata.pkl")
    with open(cal_metadata_path, "wb") as f:
        pickle.dump(cal_metadata, f)
    logging.info(f"Saved: {cal_metadata_path}")

if __name__ == "__main__":
    main()



