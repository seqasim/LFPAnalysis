import pandas as pd 
import numpy as np 
from itertools import chain

def extract_names_connect_table(connect_table_path):
    """
    Utility function for extracting channel types from Iowa connection table
    """

    connect_table = pd.read_csv(connect_table_path)

    # Strip spaces from column headers if they have them: 
    connect_table.rename(columns=lambda x: x.strip(), inplace=True)

    connect_table['Contact Location'] = connect_table['Contact Location'].str.split().str.join(' ')

    # strip \xa0 from all strings in all columns, all rows 

    connect_table.dropna(subset=['Code'], inplace=True)

    eegCode =['scalp']
    all_eeg = [x[7:].split(', ') for x in connect_table[connect_table.Code.str.lower().str.contains(eegCode[0])]['Contact Location'].astype(str).tolist()]
    eeg_labels = [x.lower().replace(u'\xa0', u' ') for x in list(chain(*all_eeg))]
    # NOTE: The following names are set MANUALLY upon data UPLOAD. In the original table they read as "BP" which is not informative.
    respCode = ['CAN', 'THERM', 'BELT']
    ekgCode = ['EKG']
    unusedCode = ['UNUSED']
    refCode = ['REF']
    sync_name = 'ttl' 

    # The relevant channels could vary in length: 
    seeg_chs = [] 

    # relevant_rows = connect_table['NLX-LFPx channel'][~connect_table.Code.isin(respCode+ekgCode+eegCode+unusedCode+refCode)].dropna()
    mask = pd.notna(connect_table['Contact Location']) & connect_table['Contact Location'].str.startswith(('Left', 'Right'))
    relevant_rows = connect_table[mask]['NLX-LFPx channel'].dropna()
    # single channels 
    seeg_chs += relevant_rows[~relevant_rows.str.contains(':')].tolist()
    # channel range:
    starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int)
    ends = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1
    for a,b in zip(starts, ends):
        seeg_chs += np.arange(a, b).tolist()
    seeg_names = [f'LFPx{ch}'.lower() for ch in seeg_chs]

    # The relevant channels could vary in length: 
    resp_chs = [] 
    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(respCode)].dropna()
    # single channels 
    resp_chs += relevant_rows[~relevant_rows.str.contains(':')].tolist()
    # channel range:
    starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int)
    ends = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1
    for a,b in zip(starts, ends):
        resp_chs += np.arange(a, b).tolist()
    resp_names = [f'LFPx{ch}'.lower() for ch in resp_chs]

    # The relevant channels could vary in length: 
    ekg_chs = [] 
    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(ekgCode)].dropna()
    # single channels 
    ekg_chs += relevant_rows[~relevant_rows.str.contains(':')].tolist()
    # channel range:
    starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int)
    ends = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1
    for a,b in zip(starts, ends):
        ekg_chs += np.arange(a, b).tolist()
    ekg_names = [f'LFPx{ch}'.lower() for ch in ekg_chs]

    # The relevant channels could vary in length: 
    eeg_chs = [] 
    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(eegCode)].dropna()
    # single channels 
    eeg_chs += relevant_rows[~relevant_rows.str.contains(':')].tolist()
    # channel range:
    starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int)
    ends = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1
    for a,b in zip(starts, ends):
        eeg_chs += np.arange(a, b).tolist()

    eeg_names = [f'LFPx{ch}'.lower() for ch in eeg_chs]

    # The relevant channels could vary in length: 
    drop_chs = [] 
    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(unusedCode)].dropna()
    # single channels 
    drop_chs += relevant_rows[~relevant_rows.str.contains(':')].tolist()
    # channel range:
    starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int)
    ends = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1
    for a,b in zip(starts, ends):
        drop_chs += np.arange(a, b).tolist()
    drop_names = [f'LFPx{ch}'.lower() for ch in drop_chs]

    return eeg_names, resp_names, ekg_names, seeg_names, drop_names

# def rename_mne_channels(mne_data, connect_table_path):
#     """ 
#     """ 

#     connect_table = pd.read_csv(connect_table_path)

#     mask = pd.notna(connect_table['Contact Location']) & connect_table['Contact Location'].str.startswith(('Left', 'Right'))
#     seeg_table = connect_table[mask].dropna()


#     mapping_name = {f'{x}': np.nan for x in mne_data.ch_names}

#     for code in seeg_table.Code.unique():
#         relevant_rows = seeg_table[seeg_table.Code==code]['NLX-LFPx channel']
#         starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int).values
#         ends = (relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1).values
#         channel_count = 1
#         for a,b in zip(starts, ends): 
#             channels = np.arange(a,b)
#             for channel in channels:
#                 mapping_name[f'lfpx{channel}'] = f'{code}_{channel_count}'
#                 channel_count += 1

#     return mapping_name

def rename_men_channels(mne_data, location_table_path):
    """
    """

    location_table = pd.read_csv(location_table_path)
    