import pandas as pd 
import numpy as np 


def extract_names_connect_table(connect_table_path):
    """
    Utility function for extracting channel types from Iowa connection table
    """

    connect_table = pd.read_csv(connect_table_path)

    eegCode =['SCALP']
    eeg_labels = [x.lower() for x in connect_table[connect_table.Code==eegCode[0]]['Contact Location'].tolist()[0][7:].split(', ')]
    # NOTE: The following names are set MANUALLY upon data UPLOAD. In the original table they read as "BP" which is not informative.
    respCode = ['CAN', 'THERM', 'BELT']
    ekgCode = ['EKG']
    unusedCode = ['UNUSED']
    refCode = ['REF']
    sync_name = 'ttl' 

    # The relevant channels could vary in length: 
    seeg_chs = [] 
    relevant_rows = connect_table['NLX-LFPx channel'][~connect_table.Code.isin(respCode+ekgCode+eegCode+unusedCode+refCode)].dropna()
    # single channels 
    seeg_chs += relevant_rows[~relevant_rows.str.contains(':')].tolist()
    # channel range:
    starts = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[0]).astype(int)
    ends = relevant_rows[relevant_rows.str.contains(':')].apply(lambda x: x.split(':')[1]).astype(int) + 1
    for a,b in zip(starts, ends):
        seeg_chs += np.arange(a, b).tolist()
    seeg_names = [f'LFPx{ch}' for ch in seeg_chs]

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
    resp_names = [f'LFPx{ch}' for ch in resp_chs]

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
    ekg_names = [f'LFPx{ch}' for ch in ekg_chs]

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

    eeg_names = [f'LFPx{ch}' for ch in eeg_chs]

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
    drop_names = [f'LFPx{ch}' for ch in drop_chs]

    return eeg_names, resp_names, ekg_names, seeg_names, drop_names