import re
import os
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def fix_id_columns(df, date_col="Date"):
    def parse_id_from_col(col_name):
        match = re.search(r"DPL#\(([\dA-Za-z]+)\(", col_name)
        if match:
            return match.group(1)
        return col_name

    new_columns = {}
    for c in df.columns:
        if c != date_col: # Skip renaming the date column
            new_columns[c] = parse_id_from_col(c)

    df_renamed = df.rename(columns=new_columns)
    return df_renamed

def melt_dataframe(df, value_name, date_series):
    df = df.copy()
    df['Date'] = date_series.values
    melted = df.melt(id_vars='Date', var_name='Stock', value_name=value_name)
    return melted

# where raw xlsx files lie in subfolders
data_path = r'D:\Datastream\US'
# collect all folders but not files from data_path
folder_names = [name for name in os.listdir(data_path) if len(name.split(".")) == 1]
folder_names.sort()
# folder where panel data sets are going to saved within data_path
destination_path = "Panel_Data_US"
store_at = os.path.join(data_path, destination_path)
logging.info("Creating folder for saving panel data sets.")
os.makedirs(store_at)

# iterate through folders
for folder_name in folder_names:
    logging.info(f"Starting with data processing of folder {folder_name}.")
    folder_path = os.path.join(data_path, folder_name)

    # read files
    logging.info(f"Starting to import raw xlsx files for folder {folder_name}.")
    return_index = pd.read_excel(os.path.join(folder_path, f"ri_{folder_name}.xlsx"), engine='openpyxl')
    volume       = pd.read_excel(os.path.join(folder_path, f"vo_{folder_name}.xlsx"), engine='openpyxl')    
    open_price   = pd.read_excel(os.path.join(folder_path, f'po_{folder_name}.xlsx'),  engine='openpyxl')
    high_price   = pd.read_excel(os.path.join(folder_path, f'ph_{folder_name}.xlsx'),  engine='openpyxl')
    low_price    = pd.read_excel(os.path.join(folder_path, f'pl_{folder_name}.xlsx'),  engine='openpyxl')
    close_price  = pd.read_excel(os.path.join(folder_path, f'p_{folder_name}.xlsx'),   engine='openpyxl')
    mcap         = pd.read_excel(os.path.join(folder_path, f'mv_{folder_name}.xlsx'), engine='openpyxl')
    adj_factor   = pd.read_excel(os.path.join(folder_path, f'af_{folder_name}.xlsx'), engine='openpyxl')
    unadj_price  = pd.read_excel(os.path.join(folder_path, f'up_{folder_name}.xlsx'), engine='openpyxl')
    mcap_to_bv   = pd.read_excel(os.path.join(folder_path, f'mtbv_{folder_name}.xlsx'), engine='openpyxl')
    logging.info(f"Finished importing raw xlsx files for folder {folder_name}.")

    # Delete error columns / Remove stocks where RI is unavailable. For the remaining variables, these columns are removed lated when reindexing
    logging.info(f"Removing error columns, renaming, setting formats for dates and values.")
    return_index  = return_index.loc[:, ~return_index.columns.str.startswith('#ERROR')]
    df_RI    = return_index.iloc[2:].copy()
    df_VO    = volume.iloc[2:].copy()
    df_PO    = open_price.iloc[2:].copy()
    df_PH    = high_price.iloc[2:].copy()
    df_PL    = low_price.iloc[2:].copy()
    df_P     = close_price.iloc[2:].copy()
    df_MV    = mcap.iloc[2:].copy()
    df_MTBV  = mcap_to_bv.iloc[2:].copy()
    df_AF    = adj_factor.iloc[2:].copy()
    df_UP    = unadj_price.iloc[2:].copy()

    df_RI.rename(columns={df_RI.columns[0]: "Date"}, inplace=True)
    df_VO.rename(columns={df_VO.columns[0]: "Date"}, inplace=True)
    df_PO.rename(columns={df_PO.columns[0]: "Date"}, inplace=True)
    df_PH.rename(columns={df_PH.columns[0]: "Date"}, inplace=True)
    df_PL.rename(columns={df_PL.columns[0]: "Date"}, inplace=True)
    df_P.rename(columns={df_P.columns[0]: "Date"}, inplace=True)
    df_MV.rename(columns={df_MV.columns[0]: "Date"}, inplace=True)
    df_MTBV.rename(columns={df_MTBV.columns[0]: "Date"}, inplace=True)
    df_AF.rename(columns={df_AF.columns[0]: "Date"}, inplace=True)
    df_UP.rename(columns={df_UP.columns[0]: "Date"}, inplace=True)

    df_RI["Date"]   = pd.to_datetime(df_RI["Date"])
    df_VO["Date"]   = pd.to_datetime(df_VO["Date"])
    df_PO["Date"]   = pd.to_datetime(df_PO["Date"])
    df_PH["Date"]   = pd.to_datetime(df_PH["Date"])
    df_PL["Date"]   = pd.to_datetime(df_PL["Date"])
    df_P["Date"]    = pd.to_datetime(df_P["Date"])
    df_MV["Date"]   = pd.to_datetime(df_MV["Date"])
    df_MTBV["Date"] = pd.to_datetime(df_MTBV["Date"])
    df_AF["Date"]   = pd.to_datetime(df_AF["Date"])
    df_UP["Date"]   = pd.to_datetime(df_UP["Date"])

    max_date =  min(
        [
            df_RI['Date'].max(),
            df_VO['Date'].max(),
            df_PO['Date'].max(),
            df_PH['Date'].max(),
            df_PL['Date'].max(),
            df_P['Date'].max(),
            df_MV['Date'].max(),
            df_MTBV["Date"].max(),
            df_AF['Date'].max(),
            df_UP['Date'].max()
        ]
    )

    df_RI   = df_RI[df_RI['Date'] <= max_date]
    df_VO   = df_VO[df_VO['Date'] <= max_date]
    df_PO   = df_PO[df_PO['Date'] <= max_date]
    df_PH   = df_PH[df_PH['Date'] <= max_date]
    df_PL   = df_PL[df_PL['Date'] <= max_date]
    df_P    = df_P[df_P['Date']   <= max_date]
    df_MV   = df_MV[df_MV['Date'] <= max_date]
    df_MTBV = df_MTBV[df_MTBV['Date'] <= max_date]
    df_AF   = df_AF[df_AF['Date'] <= max_date]
    df_UP   = df_UP[df_UP['Date'] <= max_date]

    df_RI.loc[:, df_RI.columns != "Date"] = df_RI.loc[:, df_RI.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_VO.loc[:, df_VO.columns != "Date"] = df_VO.loc[:, df_VO.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_PO.loc[:, df_PO.columns != "Date"] = df_PO.loc[:, df_PO.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_PH.loc[:, df_PH.columns != "Date"] = df_PH.loc[:, df_PH.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_PL.loc[:, df_PL.columns != "Date"] = df_PL.loc[:, df_PL.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_P.loc[:,  df_P.columns  != "Date"] = df_P.loc[:,  df_P.columns  != "Date"].apply(pd.to_numeric, errors="coerce")
    df_MV.loc[:, df_MV.columns != "Date"] = df_MV.loc[:, df_MV.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_MTBV.loc[:, df_MTBV.columns != "Date"] = df_MTBV.loc[:, df_MTBV.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_AF.loc[:, df_AF.columns != "Date"] = df_AF.loc[:, df_AF.columns != "Date"].apply(pd.to_numeric, errors="coerce")
    df_UP.loc[:, df_UP.columns != "Date"] = df_UP.loc[:, df_UP.columns != "Date"].apply(pd.to_numeric, errors="coerce")

    # get rid of raw column names
    df_RI    = fix_id_columns(df_RI)
    df_VO    = fix_id_columns(df_VO)
    df_PO    = fix_id_columns(df_PO)
    df_PH    = fix_id_columns(df_PH)
    df_PL    = fix_id_columns(df_PL)
    df_P     = fix_id_columns(df_P)
    df_MV    = fix_id_columns(df_MV)
    df_MTBV  = fix_id_columns(df_MTBV)
    df_AF    = fix_id_columns(df_AF)
    df_UP    = fix_id_columns(df_UP)

    # keep only columns without errors during data retrieval
    ref_cols = df_RI.columns
    df_RI  = df_RI.reindex(columns=ref_cols)
    df_VO  = df_VO.reindex(columns=ref_cols)
    df_PO  = df_PO.reindex(columns=ref_cols)
    df_PH  = df_PH.reindex(columns=ref_cols)
    df_PL  = df_PL.reindex(columns=ref_cols)
    df_P   = df_P.reindex(columns=ref_cols)
    df_MV  = df_MV.reindex(columns=ref_cols)
    df_MTBV  = df_MTBV.reindex(columns=ref_cols)
    df_AF  = df_AF.reindex(columns=ref_cols)
    df_UP  = df_UP.reindex(columns=ref_cols)

    # convert from wide to long format
    date_series = df_RI['Date']
    logging.info("Melting data frames.")
    df_PO_panel  = melt_dataframe(df_PO,'Open', date_series)
    df_PH_panel  = melt_dataframe(df_PH, 'High', date_series)
    df_PL_panel  = melt_dataframe(df_PL, 'Low', date_series)
    df_P_panel   = melt_dataframe(df_P, 'Close', date_series)
    df_VO_panel  = melt_dataframe(df_VO, 'Volume', date_series)
    df_RI_panel  = melt_dataframe(df_RI, 'ReturnIndex', date_series)
    df_MV_panel  = melt_dataframe(df_MV, 'MCAP', date_series)
    df_MTBV_panel  = melt_dataframe(df_MTBV, 'MTBV', date_series)
    df_AF_panel  = melt_dataframe(df_AF, 'AdjFactor', date_series)
    df_UP_panel  = melt_dataframe(df_UP, 'UnadjClose', date_series)

    # merge to one panel
    logging.info("Mergin data frames.")
    tmp_panel = df_PO_panel.merge(df_PH_panel,  on=['Date', 'Stock']) \
                            .merge(df_PL_panel,  on=['Date', 'Stock']) \
                            .merge(df_P_panel,   on=['Date', 'Stock']) \
                            .merge(df_VO_panel,  on=['Date', 'Stock']) \
                            .merge(df_RI_panel,  on=['Date', 'Stock']) \
                            .merge(df_MV_panel,  on=['Date', 'Stock']) \
                            .merge(df_MTBV_panel,  on=['Date', 'Stock']) \
                            .merge(df_AF_panel,  on=['Date', 'Stock']) \
                            .merge(df_UP_panel,  on=['Date', 'Stock'])

    logging.info("Removing stocks with no data for OHLCV and ReturnIndex.")
    missing_data_stocks = (
        tmp_panel.groupby('Stock')[['Open','High','Low','Close','Volume', 'ReturnIndex']].apply(lambda sub: sub.isna().all().any())
    )

    bad_stocks  = missing_data_stocks[missing_data_stocks].index
    OHLCV_panel = tmp_panel[~tmp_panel['Stock'].isin(bad_stocks)]
    logging.info("Saving panel data set.")
    OHLCV_panel.to_feather(os.path.join(store_at, f"OHLCV_panel_{folder_name}.feather"))
    logging.info(f"Finished with data processing of folder {folder_name}.")