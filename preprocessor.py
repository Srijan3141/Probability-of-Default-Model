import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pygam import LogisticGAM, s
from functools import reduce
import operator
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import roc_curve, roc_auc_score
from pygam import LogisticGAM, s
from functools import reduce
import operator
import numpy as np


def preprocessor(df):
    #Imputation using financial accounting equations
    def fin_acc(df):
        df['prof_operations'] = np.where(
            df['prof_operations'].isnull() & df['rev_operating'].notnull() & df['COGS'].notnull(),
            df['rev_operating'] - df['COGS'],
            df['prof_operations']
        )

        df['prof_operations'] = np.where(
            df['prof_operations'].isnull() & df['roa'].notnull() & df['asst_tot'].notnull(),
            (df['roa'] / 100) * df['asst_tot'],
            df['prof_operations']
        )

        df['rev_operating'] = np.where(
            df['rev_operating'].isnull() & df['prof_operations'].notnull() & df['COGS'].notnull(),
            df['prof_operations'] + df['COGS'],
            df['rev_operating']
        )
        return df
    
    df = fin_acc(df)

    ####Create ratios
    # Handles NA, inf,-inf logic
    def safe_divide(x, y):
        x = x.astype(float)
        y = y.astype(float)

        with np.errstate(divide='ignore', invalid='ignore'):
            out = x / y
        return out

    # Create financial ratios
    def create_ratios(df): 
        df["current_ratio"] = safe_divide(df["asst_current"], df["debt_st"])
        df['Adj_NetIncome_to_Assets'] = safe_divide((df['profit'] - df['inc_extraord']) , df['asst_tot'])
        df['asset_turnover_ratio'] = safe_divide(df['rev_operating'],df['asst_tot'])
        df["interest_expense_after_taxes"] = df["prof_operations"] - df["profit"]
        df['tie'] = safe_divide(df['prof_operations'], df['interest_expense_after_taxes'])
        df['liabilities_total'] = df['asst_tot'] - df['eqty_tot']
        df['Liabilities_TangibleAssets_Ratio'] = safe_divide(df['liabilities_total'], df['asst_tang_fixed'])
        df["ocf_ratio"] = safe_divide(df["cf_operations"], df["debt_st"])
        return df
    
    df = create_ratios(df)

    #The ratios we decided to use
    variables_used = ["current_ratio",
                    "Adj_NetIncome_to_Assets",
                    "asset_turnover_ratio",
                    "tie",     
                    "cash_and_equiv",
                    "eqty_tot",               
                    "Liabilities_TangibleAssets_Ratio",
                    "ocf_ratio"  ]

    # Wincorizing parameters
    wincor_params = {
        'ocf_ratio': {'lower_clip_value': -3.9552445122503346, 'upper_clip_value': 6.418360606381166},
        'current_ratio': {'lower_clip_value': 0.0006815897431974335, 'upper_clip_value': 31.209559605154542},
        'Adj_NetIncome_to_Assets': {'lower_clip_value': -0.37986630259527054, 'upper_clip_value': 0.2920304598106665},
        'asset_turnover_ratio': {'lower_clip_value': -0.14910570238701049, 'upper_clip_value': 4.512680994613375},
        'Liabilities_TangibleAssets_Ratio': {'lower_clip_value': 0.002695195329479733, 'upper_clip_value': 1299.7651802237524},
        'tie': {'lower_clip_value': -42.64902854750117, 'upper_clip_value': 35.57444538625191},
        'cash_and_equiv': {'lower_clip_value': -96212.038, 'upper_clip_value': 5808024.340000009},
        'eqty_tot': {'lower_clip_value': -921694.42, 'upper_clip_value': 65389330.5800003}
    }

    # Wincorize function
    def winsorize_with_bounds(df, wincor_params):
        df_out = df.copy()
        for col, params in wincor_params.items():
            lower_val = params["lower_clip_value"]
            upper_val = params["upper_clip_value"]

            s = df_out[col]

            # Replace infinities with the appropriate bound
            s = s.replace(np.inf, upper_val)
            s = s.replace(-np.inf, lower_val)

            # Clip to bounds
            s = s.clip(lower_val, upper_val)

            df_out[col] = s

        return df_out
    
    df = winsorize_with_bounds(df, wincor_params)

    # Load median imputer metadata
    with open("median_imputer_by_legal_struct_ateco_sector.pkl", "rb") as f:
        median_metadata = pickle.load(f)

    # Perform hierarchical median imputation using:
    def hierarchical_median_impute(df, ratio_cols, median_metadata):

        # Extract metadata
        granular_medians = median_metadata["granular_medians"]
        ateco_medians    = median_metadata["ateco_medians"]
        legal_medians    = median_metadata["legal_medians"]
        global_medians   = median_metadata["global_medians"]

        df = df.copy()

        for col in ratio_cols:

            missing_mask = df[col].isna()

            # Loop only through the missing rows for this column
            for idx in df[missing_mask].index:

                ateco = df.at[idx, "ateco_sector"]
                legal = df.at[idx, "legal_struct"]
                key2  = (ateco, legal)

                # 1) Full granular match
                if key2 in granular_medians.index:
                    df.at[idx, col] = granular_medians.loc[key2, col]
                    continue

                # 2) ATECO-only match
                if ateco in ateco_medians.index:
                    df.at[idx, col] = ateco_medians.loc[ateco, col]
                    continue

                # 3) Legal-struct-only match
                if legal in legal_medians.index:
                    df.at[idx, col] = legal_medians.loc[legal, col]
                    continue

                # 4) Global fallback
                df.at[idx, col] = global_medians[col]

        return df

    df = hierarchical_median_impute(df, variables_used, median_metadata)

    return df
