import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from filter import DSPreprocess
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_path      = r"/data/Datastream/PriceData/US"
save_dir = "Filtered_Data"

penny_percentile = 0.10
std_extreme_removal_threshold = 10

try:
    os.makedirs(os.path.join(data_path, save_dir))
except:
    logging.info("Saving directory seems to exist already.")

nof_subfolders = 32 

timeseries_dfs = []
static_dfs     = []

for i in tqdm(range(1, nof_subfolders + 1), desc="Load data"):
    folder_nbr       = f"{i:03d}"
    folder_path      = os.path.join(data_path, folder_nbr)
    OHLCV_panel_iter = pd.read_feather(os.path.join(data_path, "Panel_Data_US", f"OHLCV_panel_{folder_nbr}.feather"))
    static_iter      = pd.read_excel(os.path.join(folder_path, f'static_{folder_nbr}.xlsx'), engine='openpyxl')
    timeseries_dfs.append(OHLCV_panel_iter)
    static_dfs.append(static_iter)

statics = pd.concat(static_dfs, axis=0, ignore_index=True)
statics.reset_index(drop=True, inplace=True)

delist_str = statics["ENAME"].str.extract(r"DELIST\.(\d{2}/\d{2}/\d{2})")[0]
statics["Delisting Date"] = pd.to_datetime(delist_str, format="%d/%m/%y", errors="coerce")

statics['BDATE'] = pd.to_datetime(statics['BDATE'])
string_columns = ['DSCD', 'ENAME', 'EXMNEM', 'GEOGN', 'ISIN', 'ISINID', 'LOC', 'PCUR', 'TRAC', 'TYPE', 'CURRENCY']
if 'Type' in statics.columns:
    string_columns.insert(0, 'Type')

statics[string_columns] = statics[string_columns].astype(str)

OHLCV_panel = pd.concat(timeseries_dfs, axis=0, ignore_index=True)
OHLCV_panel.sort_values(by=["Date", "Stock"], inplace=True)
OHLCV_panel.reset_index(drop=True, inplace=True)

logging.info(f"Number of rows before removing duplicate Stock-Date observations: {OHLCV_panel.shape[0]}")
OHLCV_panel = OHLCV_panel.drop_duplicates(subset=["Stock", "Date"], keep="first")
logging.info(f"Number of rows after removing duplicate Stock-Date observations: {OHLCV_panel.shape[0]}")

OHLCV_panel.loc[:, 'Stock'] = OHLCV_panel['Stock'].astype(str)
OHLCV_panel.loc[:, 'Stock'] = OHLCV_panel['Stock'].str.strip()
statics.loc[:, 'DSCD']      = statics['DSCD'].str.strip()

mask = OHLCV_panel["ReturnIndex"] < 1e-6
logging.info(f"Frequency of ReturnIndex observations with extreme small values: {mask.sum()/OHLCV_panel.shape[0]:.6f}")

OHLCV_panel.loc[mask, "ReturnIndex"] = np.nan

OHLCV_panel["Return"] = (
    OHLCV_panel.groupby("Stock")["ReturnIndex"]
    .transform(lambda x: x / x.shift(1) - 1)
)

logging.info(f"Number of companies before removing non-regional companies: {statics.shape[0]}")
statics = statics[statics['GEOGN'] == 'UNITED STATES']
logging.info(f"Number of companies after removing non-regional companies: {statics.shape[0]}")

########################################################################################################################
## Filters based on static data
########################################################################################################################
# Filter (1) - Equity filter:
OHLCV_panel = DSPreprocess.filter_non_common_stocks(OHLCV_panel, statics, country='UNITED STATES')


# Filter (2) - Cross-listing filter:
OHLCV_panel = DSPreprocess.filter_cross_listings(OHLCV_panel, statics, country='UNITED STATES')


# Filter (3): Duplicate LOC Codes
OHLCV_panel = DSPreprocess.filter_duplicate_loc_codes(OHLCV_panel, statics)


# Filter (4) - Foreign firms:
OHLCV_panel = OHLCV_panel[OHLCV_panel.Stock.isin(statics.DSCD.unique())]


# Filter (5) - Stocks in foreign currencies:
OHLCV_panel = DSPreprocess.filter_foreign_currency_stocks(OHLCV_panel, statics, country='UNITED STATES')


# Filter (17) - Survivorship bias (obsolete for US data, because our dataset starts in 1993):
# OHLCV_panel = DSPreprocess.filter_surivorship_bias(OHLCV_panel, statics, country='UNITED STATES')


########################################################################################################################
## Filters based on ReturnIndex
########################################################################################################################
# Filter (7) - :
# Remove stocks of which more than 98% of non-zero mean returns are either positive or negative
OHLCV_panel = DSPreprocess.filter_implausible_returns(OHLCV_panel)

########################################################################################################################
## Stockday filters:
########################################################################################################################
# Filter (13):
# If RI is forward filled for 10 consecutive days, then remove those days.
OHLCV_panel = DSPreprocess.filter_padded_values_delistings(OHLCV_panel, statics)


# Filter (8):
# Remove stocks for which the returns are zero in more than 95% of their sample (After applying filter (13).
OHLCV_panel = DSPreprocess.filter_zero_return_stocks(OHLCV_panel)


# Filter (14):
# Stale prices
OHLCV_panel = DSPreprocess.filter_stale_prices(OHLCV_panel)


# Filter (9):
# Remove stocks with a daily standard deviation of more than 40%.
OHLCV_panel  = DSPreprocess.filter_stocks_by_high_volatility(OHLCV_panel, volatility_threshold=0.40)


# Filter (10):
# Remove stocks with a daily standard deviation of less than 0.01 bps.
OHLCV_panel = DSPreprocess.filter_stocks_by_low_volatility(OHLCV_panel)


# Filter (15):
# Target filter rate not reported / ~0.0015% (~0.00569% when applied on raw panel) actual filter rate
OHLCV_panel = DSPreprocess.filter_outlier_errors(OHLCV_panel, up_ts=1.0, down_ts=-0.5, method='drop')


# Filter (16):
# Holiday filter: Has to be applied after filter (11) and (13)!
# Remove days on which non-missing or non-zero returns account for less than 0.5% of total available stocks.
OHLCV_panel = DSPreprocess.filter_holidays(OHLCV_panel)


# Filter (Own - implausible OHLC):
# Nonsense values (Low > (Open OR High OR Close) and High < (Open OR Low OR Close):
OHLCV_panel = DSPreprocess.filter_implausible_prices(OHLCV_panel)


# Filter (Extreme prices - Schmidt, von Arx (2011))
# Remove prices higher than 1mio US$.
# OHLCV_panel = DSPreprocess.filter_extreme_prices(OHLCV_panel, ts=1_000_000)


# NOT NEEDED - Filter (20 - Extreme returns due to decimal errors - Annaert et al. (2013) JBF)
# OHLCV_panel = DSPreprocess.filter_decimal_errors(OHLCV_panel, up_ts=4.0, down_ts=-0.85)


# Filter (No trading activity - Chaieb et al. (2021) JoFE)
OHLCV_panel = DSPreprocess.filter_no_trading_activity(OHLCV_panel)

#
# # Filter (Own - Extreme returns)
# # OHLCV_panel = DSPreprocess.filter_extreme_returns(OHLCV_panel, lower=0.00, upper=0.999)
# OHLCV_panel = DSPreprocess.filter_extreme_returns2(OHLCV_panel, n_std=5) # Less aggressive than above version.


# Filter (Own - NA filter) - Drop all rows before they are populated for the first time and apply forward + backward fill.
OHLCV_panel = DSPreprocess.handle_missings(OHLCV_panel, statics, country='UNITED STATES')


########################################################################################################################
# Manual removal of implausibilities:
########################################################################################################################
# 1.) AgEagle Aerial Systems, Inc. (680683). Remove observations before foundation date.
OHLCV_panel = OHLCV_panel[~((OHLCV_panel['Stock'] == "680683") & (OHLCV_panel['Date'] < '2010-01-01'))]


# 2.) Strange prices due to stock splits / reverse splits
OHLCV_panel = OHLCV_panel[~(OHLCV_panel["Stock"]  == "872328")]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "9364PF") & (OHLCV_panel["Date"] == "2020-07-13"))]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "50259R") & (OHLCV_panel["Date"] == "2018-01-04"))]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "67684T") & (OHLCV_panel["Date"] == "2023-08-07"))]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "2566DU") & (OHLCV_panel["Date"] == "2024-06-04"))]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "28355P") & (OHLCV_panel["Date"] == "2008-12-17"))]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "2634G3") & (OHLCV_panel["Date"] == "2024-03-27"))]
OHLCV_panel = OHLCV_panel[~((OHLCV_panel["Stock"] == "32650J") & (OHLCV_panel["Date"] == "2009-05-07"))]


# 3.) Implausible returns due to high amount of missings:
OHLCV_panel = OHLCV_panel[~(OHLCV_panel["Stock"] == "7076TJ")]
OHLCV_panel = OHLCV_panel[~(OHLCV_panel["Stock"] == "7076TK")]


# 4.) Seems to be some sort of dividend split, that was incorrectly labeled as a stock:
OHLCV_panel = OHLCV_panel[~(OHLCV_panel["Stock"] == "92238K")]


# 5.) Delete day with unusual drop in listed firms:
# OHLCV_panel = OHLCV_panel[~(OHLCV_panel["Date"] == '1995-05-26')]

# Filter (18) - Adjustment inconsistencies.
OHLCV_panel_temp = DSPreprocess.filter_adjustment_inconsistencies(OHLCV_panel, threshold=0.05)


# Filter (21) - Penny stocks.
OHLCV_panel, penny_thresholds = DSPreprocess.filter_penny_stocks(OHLCV_panel, 0.10)
logging.info("Saving monthly thresholds for penny stock selection.")
penny_thresholds.to_csv(os.path.join(data_path, save_dir, "penny_stock_thresholds.csv"), index = True)

# Filter (Own - Extreme returns)
# OHLCV_panel = DSPreprocess.filter_extreme_returns(OHLCV_panel, lower=0.00, upper=0.999)
OHLCV_panel = DSPreprocess.filter_extreme_returns2(OHLCV_panel, n_std=10) # Less aggressive than above version.


# Filter (12) - Filters the panel to include only stocks with sufficient observation history
OHLCV_panel = DSPreprocess.filter_short_history_stocks(OHLCV_panel, threshold=120)


OHLCV_panel.replace([np.inf, -np.inf], np.nan, inplace=True)
OHLCV_panel_final = OHLCV_panel.dropna(subset=['Return'])

# extract companies which are in the final filtered data set
statics_for_filtered = statics[statics.DSCD.isin(OHLCV_panel_final.Stock.unique().tolist())]

# save
logging.info("Saving data and static information for remaining companies.")
OHLCV_panel_final.to_feather(os.path.join(data_path, save_dir, f"Financial_base_data_panel_filtered_{penny_percentile}_{std_extreme_removal_threshold}.feather"))
statics_for_filtered.to_csv(os.path.join(data_path, save_dir, f"statics_filtered_{penny_percentile}_{std_extreme_removal_threshold}.csv"), index = False)