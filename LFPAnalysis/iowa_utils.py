import pandas as pd 
import numpy as np 
from LFPAnalysis import lfp_preprocess_utils


def _expand_channel_rows(channel_rows: pd.Series) -> list[int]:
    """Expand single-channel and ranged Iowa channel entries into integers."""

    if channel_rows.empty:
        return []

    normalized_rows = channel_rows.astype(str).str.strip()
    single_mask = ~normalized_rows.str.contains(':')
    channels = normalized_rows[single_mask].astype(int).tolist()

    range_rows = normalized_rows[~single_mask]
    if not range_rows.empty:
        bounds = range_rows.str.split(':', expand=True).astype(int).to_numpy()
        channels.extend(
            np.concatenate([np.arange(start, end + 1) for start, end in bounds]).tolist()
        )

    return channels


def _to_lfpx_names(channels) -> list[str]:
    """Format integer Neuralynx channels as canonical lowercase names."""

    return [f'LFPx{channel}'.lower() for channel in channels]


def extract_names_connect_table(connect_table_path: str):
    """Extract channel types from Iowa connection table.
    
    Parameters
    ----------
    connect_table_path : str
        Path to the connection table CSV file.
    
    Returns
    -------
    tuple
        Tuple containing (eeg_names, resp_names, ekg_names, seeg_names, drop_names).
    """

    connect_table = pd.read_csv(connect_table_path)

    # Strip spaces from column headers if they have them: 
    connect_table.rename(columns=lambda x: x.strip(), inplace=True)

    connect_table['Contact Location'] = connect_table['Contact Location'].str.split().str.join(' ')

    # strip \xa0 from all strings in all columns, all rows 

    connect_table.dropna(subset=['Code'], inplace=True)

    eegCode =['scalp']
    # NOTE: The following names are set MANUALLY upon data UPLOAD. In the original table they read as "BP" which is not informative.
    respCode = ['CAN', 'THERM', 'BELT']
    ekgCode = ['EKG']
    unusedCode = ['UNUSED']

    # relevant_rows = connect_table['NLX-LFPx channel'][~connect_table.Code.isin(respCode+ekgCode+eegCode+unusedCode+refCode)].dropna()
    mask = pd.notna(connect_table['Contact Location']) & connect_table['Contact Location'].str.startswith(('Left', 'Right'))
    relevant_rows = connect_table[mask]['NLX-LFPx channel'].dropna()
    seeg_names = _to_lfpx_names(_expand_channel_rows(relevant_rows))

    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(respCode)].dropna()
    resp_names = _to_lfpx_names(_expand_channel_rows(relevant_rows))

    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(ekgCode)].dropna()
    ekg_names = _to_lfpx_names(_expand_channel_rows(relevant_rows))

    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(eegCode)].dropna()
    eeg_names = _to_lfpx_names(_expand_channel_rows(relevant_rows))

    relevant_rows = connect_table['NLX-LFPx channel'][connect_table.Code.isin(unusedCode)].dropna()
    drop_names = _to_lfpx_names(_expand_channel_rows(relevant_rows))

    return eeg_names, resp_names, ekg_names, seeg_names, drop_names

def extract_names_elec_table(elec_table_path: str):
    """Extract channel names from electrode table.
    
    Parameters
    ----------
    elec_table_path : str
        Path to the electrode table file.
    
    Returns
    -------
    list
        List of sEEG channel names.
    """

    elec_data = lfp_preprocess_utils.load_elec(elec_table_path, site='UI')

    seeg_chs = elec_data[elec_data.ElectrodeType.isin(['Depth', 'Subdural'])].Channel.values

    seeg_names = [f'LFPx{ch}'.lower() for ch in seeg_chs]

    return seeg_names

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

def rename_mne_channels(mne_data, location_table_path: str):
    """Rename MNE channels based on location table.
    
    Parameters
    ----------
    mne_data
        MNE data object.
    location_table_path : str
        Path to the location table CSV file.
    """
    location_table = pd.read_csv(location_table_path)
    
